import torch

from alphafold.data.feature_extraction import create_features_from_a3m
from alphafold.utils.utils import load_openfold_weights
from alphafold.model.model import Model

from alphafold.utils.utils import to_modelcif
from alphafold.utils.residue_constants import restypes

WEIGHTS_PATH = "/home/ahmedselimuzum/jff/jff_alphafold/weights/finetuning_2.pt"
A3M_PATH = "alignment_tautomerase.a3m"
N_RECYCLE = 4

if torch.cuda.is_available():
    device = 'cuda'
    print('Compatible GPU available.')
elif torch.backends.mps.is_available():
    device = 'mps'
    print('MPS (Metal Performance Shaders) available.')
else:
    device = 'cpu'
    print('No compatible GPU, fallback to CPU.')

model = Model()
openfold_weights = load_openfold_weights(WEIGHTS_PATH)

model.load_state_dict(openfold_weights)

batch = {}

cycle_batches = []
for i in range(N_RECYCLE):
    batch = create_features_from_a3m(A3M_PATH)
    cycle_batches.append(batch)

for key in cycle_batches[0].keys():
    batch[key] = torch.stack([cycle[key] for cycle in cycle_batches], dim=-1)

model.to(device)
for key, value in batch.items():
    batch[key] = value.to(device)

with torch.no_grad():
    outputs = model(batch)
