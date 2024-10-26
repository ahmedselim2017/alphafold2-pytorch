import math
import torch

testdata_dir = "tests/testdata/feature_extraction_control_values"


def test_load_a3m():
    from alphafold.data.feature_extraction import load_a3m

    seqs = load_a3m(f"{testdata_dir}/alignment_tautomerase.a3m")

    first_expected = [
        'PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK',
        'PVVTIELWEGRTPEQKRELVRAVSSAISRVLGCPEEAVHVILHEVPKANWGIGGRLASE',
        'PVVTIEMWEGRTPEQKKALVEAVTSAVAGAIGCPPEAVEVIIHEVPKVNWGIGGQIASE',
        'PIIQVQMLKGRSPELKKQLISEITDTISRTLGSPPEAVRVILTEVPEENWGVGGVPINE',
        'PFVQIHMLEGRTPEQKKAVIEKVTQALVQAVGVPASAVRVLIQEVPKEHWGIGGVSARE'
    ]
    assert len(seqs) == 8361 and seqs[:5] == first_expected


def test_onehot_encode_aa():
    from alphafold.data.feature_extraction import onehot_encode_aa

    test_seq = "ARNDCQEGHILKMFPSTWYV"

    enc1 = onehot_encode_aa(test_seq, use_gap_token=False)
    enc2 = onehot_encode_aa(test_seq, use_gap_token=True)
    enc3 = onehot_encode_aa(test_seq + '-', use_gap_token=True)

    assert torch.allclose(
        enc1, torch.nn.functional.one_hot(torch.arange(20), num_classes=21))
    assert torch.allclose(
        enc2, torch.nn.functional.one_hot(torch.arange(20), num_classes=22))
    enc3_exp = torch.nn.functional.one_hot(torch.cat(
        (torch.arange(20), torch.tensor([21]))),
                                           num_classes=22)
    assert torch.allclose(enc3, enc3_exp)


def test_initial_data_from_seqs():
    from alphafold.data.feature_extraction import load_a3m, initial_data_from_seqs

    seqs = load_a3m(f'{testdata_dir}/alignment_tautomerase.a3m')

    features = initial_data_from_seqs(seqs)

    expected_features = torch.load(f'{testdata_dir}/initial_data.pt',
                                   weights_only=False)

    for key, param in features.items():
        assert torch.allclose(
            param,
            expected_features[key]), f'Error in computation of feature {key}.'


def test_select_cluster_centers():
    from alphafold.data.feature_extraction import select_cluster_centers
    inp = torch.load(f'{testdata_dir}/initial_data.pt', weights_only=False)

    features = select_cluster_centers(inp, seed=0)

    expected_features = torch.load(f'{testdata_dir}/clusters_selected.pt',
                                   weights_only=False)

    for key, param in features.items():
        assert torch.allclose(
            param,
            expected_features[key]), f'Error in computation of feature {key}.'


def test_mask_cluster_centers():
    from alphafold.data.feature_extraction import mask_cluster_centers
    inp = torch.load(f'{testdata_dir}/clusters_selected.pt',
                     weights_only=False)

    features = mask_cluster_centers(inp, seed=1)

    expected_features = torch.load(f'{testdata_dir}/clusters_masked.pt',
                                   weights_only=False)

    for key, param in features.items():
        assert torch.allclose(
            param,
            expected_features[key]), f'Error in computation of feature {key}.'


def test_cluster_assignment():
    from alphafold.data.feature_extraction import cluster_assignment
    inp = torch.load(f'{testdata_dir}/clusters_masked.pt', weights_only=False)

    features = cluster_assignment(inp)

    expected_features = torch.load(f'{testdata_dir}/clusters_assigned.pt',
                                   weights_only=False)

    for key, param in features.items():
        assert torch.allclose(
            param,
            expected_features[key]), f'Error in computation of feature {key}.'


def test_cluster_average():
    from alphafold.data.feature_extraction import cluster_average

    N_clust = 10
    N_res = 3
    N_extra = 20
    dim1 = 5
    dim2 = 7
    assignment = torch.tensor(
        [7, 1, 1, 8, 3, 4, 7, 1, 4, 4, 9, 8, 4, 8, 1, 5, 8, 8, 8, 5])
    assignment_count = torch.tensor([0, 4, 0, 1, 4, 2, 0, 2, 6, 1])

    ft1_shape = (N_clust, N_res, dim1)
    eft1_shape = (N_extra, N_res, dim1)
    ft2_shape = (N_clust, N_res, dim1, dim2)
    eft2_shape = (N_extra, N_res, dim1, dim2)

    ft1 = torch.linspace(-2, 2, math.prod(ft1_shape)).reshape(ft1_shape)
    eft1 = torch.linspace(-2, 2, math.prod(eft1_shape)).reshape(eft1_shape)
    ft2 = torch.linspace(-2, 2, math.prod(ft2_shape)).reshape(ft2_shape)
    eft2 = torch.linspace(-2, 2, math.prod(eft2_shape)).reshape(eft2_shape)

    res1 = cluster_average(ft1, eft1, assignment, assignment_count)
    res2 = cluster_average(ft2, eft2, assignment, assignment_count)

    expected_res1 = torch.load(f'{testdata_dir}/cluster_average_res1.pt',
                               weights_only=False)
    expected_res2 = torch.load(f'{testdata_dir}/cluster_average_res2.pt',
                               weights_only=False)

    assert torch.allclose(res1, expected_res1)
    assert torch.allclose(res2, expected_res2)


def test_summarize_clusters():
    from alphafold.data.feature_extraction import summarize_clusters
    inp = torch.load(f'{testdata_dir}/clusters_assigned.pt',
                     weights_only=False)

    features = summarize_clusters(inp)

    expected_features = torch.load(f'{testdata_dir}/clusters_summarized.pt',
                                   weights_only=False)

    for key, param in features.items():
        assert torch.allclose(
            param,
            expected_features[key]), f'Error in computation of feature {key}.'


def test_crop_extra_msa():
    from alphafold.data.feature_extraction import crop_extra_msa
    inp = torch.load(f'{testdata_dir}/clusters_summarized.pt',
                     weights_only=False)

    features = crop_extra_msa(inp, seed=2)

    expected_features = torch.load(f'{testdata_dir}/extra_msa_cropped.pt',
                                   weights_only=False)

    for key, param in features.items():
        assert torch.allclose(
            param,
            expected_features[key]), f'Error in computation of feature {key}.'


def test_msa_feat():
    from alphafold.data.feature_extraction import calculate_msa_feat
    inp = torch.load(f'{testdata_dir}/extra_msa_cropped.pt',
                     weights_only=False)

    msa_feat = calculate_msa_feat(inp)

    expected_feat = torch.load(f'{testdata_dir}/msa_feat.pt',
                               weights_only=False)

    print(msa_feat.shape)
    print(expected_feat.shape)
    assert torch.allclose(msa_feat, expected_feat)


def test_calculate_extra_msa_feat():
    from alphafold.data.feature_extraction import calculate_extra_msa_feat
    inp = torch.load(f'{testdata_dir}/extra_msa_cropped.pt',
                     weights_only=False)

    msa_feat = calculate_extra_msa_feat(inp)

    expected_feat = torch.load(f'{testdata_dir}/extra_msa_feat.pt',
                               weights_only=False)

    assert torch.allclose(msa_feat, expected_feat)
