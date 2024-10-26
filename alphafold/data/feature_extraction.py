from pathlib import Path
import re

import torch
import torch.nn.functional as F

aa_alphabet = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V"
]
aa_alphabet_with_x = aa_alphabet + ["X"]
aa_alphabet_with_x_and_gap = aa_alphabet_with_x + ["-"]


def load_a3m(filename: str) -> list[str]:
    """
    Reads the given a3m MSA file and returns the sequences as a list.

    Args:
        filename:   Name of the a3m file as a string.

    Returns:
        A list of strings that has the sequences of the given a3m MSA file
        as its elements.
    """
    filepath = Path(filename)
    if not filepath.is_file():
        raise FileNotFoundError(f"The MSA file {filename} is not a file.")

    seqs: list[str] = []
    with open(filepath, "r") as fh:
        is_next_one_seq = False

        for line in fh.readlines():
            if line.startswith("#"):
                continue
            elif line.startswith(">"):
                is_next_one_seq = True
            elif is_next_one_seq:
                seqs.append(line.strip())
                is_next_one_seq = False
    return seqs


def onehot_encode_aa(seq: str, use_gap_token: bool) -> torch.Tensor:
    """
    Creates a one-hot encoding from a protein sequence.

    Args:
        seq:    The protein sequence as a string.
        use_gap_token:  Uses gap token while one-hot encoding if True.

    Returns:
        The one-hot encoding of the protein sequence in the form of a PyTorch
        tensor with a shape of (N_res, 22) if use_gap_token is True or a shape
        of (N_res, 21) if use_gap_token is False.
    """

    alphabet = aa_alphabet_with_x_and_gap if use_gap_token else aa_alphabet_with_x
    indices = [alphabet.index(aa) for aa in seq]

    return F.one_hot(torch.as_tensor(indices), num_classes=len(alphabet))


def initial_data_from_seqs(seqs: list[str]) -> dict:
    """
    Processes sequences from the a3m MSA.

    Args:
        seqs:   List of strings that has the sequences loaded from the a3m file
                where lowercase letters represent deletions.

    Returns:
        A dictionary containing:
            msa_aatype:    A PyTorch tensor of one-hot encoded sequences with
                            a shape of (N_seq, N_res, 22) where N_seq is the
                            number of unique sequences in the list seq, N_res
                            is the length of the sequences. The third dimension
                            represents the 20 possible amino acids, the unknown
                            amino acid token, and the gap token.
            msa_deletion_count:  A PyTorch tensor with a shape of
                                    (N_seq, N_res) where each element is the
                                    number of deletions happened before it.
            aa_distribution:    A PyTorch tensor with a shape of (N_res, 22)
                                that represents the distribution of each
                                amino acid in each residue location.
    """

    initial_data: dict[str, torch.Tensor] = {}

    N_res = len(seqs[0])

    unique_seq_list: list[str] = []
    deletion_count_list: list[list[int]] = []
    for seq in seqs:
        if seq.isupper() and seq not in unique_seq_list:
            unique_seq_list.append(seq)
            deletion_count_list.append([0] * N_res)
        else:
            deleted_seq = re.sub(r'[^A-Z|-]', r'', seq)
            if deleted_seq not in unique_seq_list:
                unique_seq_list.append(deleted_seq)

                seq_deletion_count_list: list[int] = [0] * N_res
                deletion_count = 0
                deleted_seq_ind = 0
                for aa in seq:
                    if aa.islower():
                        deletion_count += 1
                    else:
                        if deletion_count != 0:
                            seq_deletion_count_list[
                                deleted_seq_ind] = deletion_count
                            deletion_count = 0
                        deleted_seq_ind += 1

                deletion_count_list.append(seq_deletion_count_list)

    unique_seqs = torch.stack(
        [onehot_encode_aa(s, use_gap_token=True) for s in unique_seq_list])
    deletion_count_matrix = torch.as_tensor(deletion_count_list)

    initial_data["msa_aatype"] = unique_seqs.float()
    initial_data["msa_deletion_count"] = deletion_count_matrix.float()
    initial_data["aa_distribution"] = torch.mean(initial_data["msa_aatype"],
                                                 dim=0)

    return initial_data


def select_cluster_centers(features: dict[str, torch.Tensor],
                           max_msa_clusters=512,
                           seed: int | None = None) -> dict[str, torch.Tensor]:
    """
    Selects the cluster centers to reduce redundancy.

    Args:
        features:   A dictionary of feature representation of MSA.
        max_msa_clusters:   The maximum count of MSA cluster centers that will
                            be selected.
        seed:   Optional integer seed for the random number generator.

    Returns:
        The features dictionary modified in-place by:
            Modifying the 'msa_aatype' and 'msa_deletion_count' to contain
            data for cluster centers only.

            Adding 'extra_msa_aatype' and 'extra_msa_deletion_count' for
            non cluster center sequences.
    """

    N_seq = features['msa_aatype'].shape[0]
    if max_msa_clusters > N_seq:
        raise ValueError((f"max_msa_clusters ({max_msa_clusters}) must be",
                          "smaller or equal to the number of sequences in",
                          f"features variable({N_seq}). "))

    max_msa_clusters = min(max_msa_clusters, N_seq)

    generator = None
    if seed is not None:
        generator = torch.Generator(features["msa_aatype"].device)
        generator.manual_seed(seed)

    shuffled_i = torch.cat(
        (torch.tensor([0]),
         torch.randperm(N_seq - 1, generator=generator) + 1))

    for f_key in ['msa_aatype', 'msa_deletion_count']:

        extra_f_key = f"extra_{f_key}"
        features[extra_f_key] = features[f_key][shuffled_i[max_msa_clusters:]]
        features[f_key] = features[f_key][shuffled_i[:max_msa_clusters]]

    return features


def mask_cluster_centers(features: dict[str, torch.Tensor],
                         mask_probability=0.15,
                         seed: int | None = None) -> dict[str, torch.Tensor]:
    """
    Randomly masks the cluster centers of MSA.

    Args:
        features:   A dictionary of feature representation of clustered MSA.
        mask_probability:   Probability of masking an amino acid.
        seed:   Optional integer seed for the random number generator.

    Returns:
        The features dictionary modified in-place by:
            Modifying the 'msa_aatype' by masking.

            Copying the original 'msa_aatype' to 'true_msa_aatype'.
    """

    N_clust, N_res = features["msa_aatype"].shape[:2]

    probabs = {
        "replace_uniformly": 0.1,
        "replace_from_aa_dist": 0.1,
        "dont_replace": 0.1,
        "masked_msa_token": 0.7
    }

    gen = None
    if seed is not None:
        gen = torch.Generator(features["msa_aatype"].device)
        gen.manual_seed(seed)
        torch.manual_seed(seed)

    # probability of uniform replacement probability * 1/20 for each of the 20
    # aa and 0 for masked and unknown aa

    # (N_clust, N_res, 22)
    uniform_dist = torch.tensor([probabs["replace_uniformly"] * 1 / 20] * 20 +
                                [0, 0])
    uniform_dist = uniform_dist.broadcast_to(features["msa_aatype"].shape)

    # (N_clust, N_res, 22)
    aa_dist_dist = probabs["replace_from_aa_dist"] * features["aa_distribution"]
    aa_dist_dist = aa_dist_dist.broadcast_to(features["msa_aatype"].shape)

    # (N_clust, N_res, 22)
    dont_replace_dist = probabs["dont_replace"] * features["msa_aatype"]

    # (N_clust, N_res, 1)
    masked_dist = probabs["masked_msa_token"] * torch.ones((N_clust, N_res, 1))

    change_dist_wo_masked = uniform_dist + aa_dist_dist + dont_replace_dist
    change_dist = torch.cat((change_dist_wo_masked, masked_dist), dim=-1)

    change_with = torch.distributions.Categorical(change_dist).sample()
    change_with = F.one_hot(change_with, num_classes=23).float()

    change_mask = torch.rand((N_clust, N_res),
                             generator=gen) < mask_probability

    features["true_msa_aatype"] = features["msa_aatype"].clone()
    masked_token_padding = torch.zeros((N_clust, N_res, 1))
    features["msa_aatype"] = torch.cat(
        (features["msa_aatype"], masked_token_padding), dim=-1)

    features["msa_aatype"][change_mask] = change_with[change_mask]

    return features


def cluster_assignment(
        features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Assigns the extra sequences to the cluster centers by their Hamming
    distances to the cluster centers.

    Args:
        features: A dictionary of feature representation of clustered MSA.

    Returns:
        The features dictionary updated by the addition of:
            cluster_assignment: A PyTorch tensor of shape (N_extra, ) that
                                contains the assigned cluster centers.

            cluster_assignment_counts:  A PyTorch tensor of shape (N_clust, )
                                        containing the number of extra
                                        sequences that has been assigned to
                                        a cluster center.
    """

    # remove gap and masked tokens
    msa_aatype_filtered_aa = features["msa_aatype"][:, :, :21]
    extra_msa_aatype_filtered_aa = features["extra_msa_aatype"][:, :, :21]

    # (N_clust, N_extra)
    agreement = torch.einsum("cra,era->ce", msa_aatype_filtered_aa,
                             extra_msa_aatype_filtered_aa)

    features["cluster_assignment"] = torch.argmax(agreement, dim=0)

    features["cluster_assignment_counts"] = torch.bincount(
        features["cluster_assignment"],
        minlength=features["msa_aatype"].shape[0])
    return features


def cluster_average(cluster_center_feature: torch.Tensor,
                    extra_feature: torch.Tensor,
                    cluster_assignment: torch.Tensor,
                    cluster_assignment_count: torch.Tensor) -> torch.Tensor:
    """
    Finds the average representation for each cluster from its extra features.


    Args:
        cluster_center_feature: A PyTorch tensor of shape (N_clust, N_res, *)
                                that contains feature representations for
                                cluster centers.

        extra_feature:  A PyTorch tensor of shape (N_extra, N_res, *) that
                        contains feature representations for extra sequences.

        cluster_assignment: A PyTorch tensor of shape (N_extra,) that
                            contains the cluster assignments for each extra
                            sequence.

        cluster_assignment_count:   A PyTorch tensor of shape (N_clust,) that
                                    contains the extra sequence assignment
                                    count for each cluster.

    Returns:
        A PyTorch tensor that of shape (N_clust, N_res, *) that contains the
        average feature representations for each cluster.
    """

    N_clust = cluster_center_feature.shape[0]
    N_extra = cluster_assignment.shape[0]

    # make cluster_assignment_count broadcasted to (N_clus, N_res, *)
    bc_clust_assign_count_shape = (
        N_clust, ) + (1, ) * (cluster_center_feature.dim() - 1)

    bc_clust_assign_count = cluster_assignment_count.view(
        bc_clust_assign_count_shape).broadcast_to(cluster_center_feature.shape)

    # make cluster_assignment broadcasted to (N_extra, N_res, *)
    bc_clust_assign_shape = (N_extra, ) + (1, ) * (extra_feature.dim() - 1)
    bc_clust_assign = cluster_assignment.view(
        bc_clust_assign_shape).broadcast_to(extra_feature.shape)

    clust_sum = torch.scatter_add(cluster_center_feature,
                                  dim=0,
                                  index=bc_clust_assign,
                                  src=extra_feature)

    clust_average = clust_sum / (bc_clust_assign_count + 1)

    return clust_average


def summarize_clusters(features: dict[str, torch.Tensor]) -> dict:
    """
    Summarizes cluster by averaging MSA amino acid representations and deletion
    counts.

    Args:
        features: A dictionary of feature representation of clustered MSA.

    Returns:
        The features dictionary updated by the addition of:
            cluster_deletion_mean:  A PyTorch tensor of size (N_clust, N_res)
            that includes the average deletion counts for each cluster center

            cluster_profile:    A PyTorch tensor of size (N_clust, N_res, 23)
                                that contains average amino acid
                                representations for each cluster center.
    """

    cluster_deletion_mean = cluster_average(
        features["msa_deletion_count"], features["extra_msa_deletion_count"],
        features["cluster_assignment"], features["cluster_assignment_counts"])

    cluster_deletion_mean = (2 / torch.pi) * torch.arctan(
        cluster_deletion_mean / 3)

    pad = torch.zeros(features["extra_msa_aatype"].shape[:-1] + (1, ),
                      dtype=features["extra_msa_aatype"].dtype)

    cluster_profile = cluster_average(
        features["msa_aatype"],
        torch.cat((features["extra_msa_aatype"], pad), dim=-1),
        features["cluster_assignment"], features["cluster_assignment_counts"])

    features["cluster_deletion_mean"] = cluster_deletion_mean
    features["cluster_profile"] = cluster_profile
    return features


def crop_extra_msa(features: dict[str, torch.Tensor],
                   max_extra_msa_count=5120,
                   seed: int | None = None) -> dict[str, torch.Tensor]:
    """
    Reduces the number of extra MSAs to the max_extra_msa_count.

    Args:
        features:   A dictionary of feature representation of clustered MSA.
                    max_extra_msa_count.
        max_extra_msa_count:    Maximum number of extra sequences that should
                                be kept.
        seed:   Optional integer seed for the random number generator.

    Returns:
        The features dictionary modified in-place by cropping each key starting
        with extra_ to the desired max_extra_msa_count size.
    """

    gen = None
    if seed is not None:
        gen = torch.Generator(features['extra_msa_aatype'].device)
        gen.manual_seed(seed)

    N_extra = features["extra_msa_aatype"].shape[0]
    max_extra_msa_count = min(max_extra_msa_count, N_extra)

    kept_inds = torch.randperm(N_extra, generator=gen)[:max_extra_msa_count]

    for key in features.keys():
        if key.startswith("extra_"):
            features[key] = features[key][kept_inds, ...]
    return features


def calculate_msa_feat(features: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Concats multiple MSA features in one PyTorch tensor for structure
    prediction.

    Args:
        features:   A dictionary of feature representation of MSA.

    Returns:
        A PyTorch tensor of size (N_clust, N_res, 49) that contains msa_aatype,
        cluster_has_deletion, normalised_msa_deletion_count, cluster_profile,
        and cluster_deletion_mean.
    """

    normalised_msa_deletion_count = (2 / torch.pi) * torch.arctan(
        features["msa_deletion_count"] / 3)

    normalised_msa_deletion_count = normalised_msa_deletion_count.unsqueeze(-1)

    cluster_has_deletion = (normalised_msa_deletion_count > 0).float()

    msa_feat = torch.cat(
        (features["msa_aatype"], cluster_has_deletion,
         normalised_msa_deletion_count, features["cluster_profile"],
         features["cluster_deletion_mean"].unsqueeze(-1)),
        dim=-1)

    return msa_feat


def calculate_extra_msa_feat(
        features: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Concats multiple MSA features that start with 'extra_' in one PyTorch
    tensor for structure prediction.

    Args:
        features:   A dictionary of feature representation of MSA.


    Returns:
        A PyTorch tensor of size (N_clust, N_res, 25) that contains msa_aatype,
        normalised msa_deletion_count, cluster_deletion_mean, cluster_profile.

    """
    N_extra, N_res = features['extra_msa_aatype'].shape[:2]

    normal_extra_msa_del_count = (2 / torch.pi) * torch.arctan(
        features["extra_msa_deletion_count"] / 3)

    normal_extra_msa_del_count = normal_extra_msa_del_count.unsqueeze(-1)

    extra_msa_has_deletion = (normal_extra_msa_del_count > 0).float()

    # as the extra_msa_aatype does not have the masking token in its end
    pad = torch.zeros((N_extra, N_res, 1))

    extra_msa_feat = torch.cat(
        (features["extra_msa_aatype"], pad, extra_msa_has_deletion,
         normal_extra_msa_del_count),
        dim=-1)

    return extra_msa_feat
