import torch


def test_dropout():
    # Set this to `True` if you want to test your dropout implementation.
    from alphafold.model.dropout import DropoutRowwise, DropoutColumnwise

    test_shape = (8, 25, 30, 4)
    dropout_rowwise = DropoutRowwise(p=0.2)
    dropout_columnwise = DropoutColumnwise(p=0.3)
    dropout_rowwise.train()
    dropout_columnwise.train()

    test_inp = torch.ones(test_shape)
    rows_dropped = dropout_rowwise(test_inp)
    cols_dropped = dropout_columnwise(test_inp)

    p_nonzero_rows = torch.count_nonzero(
        rows_dropped).item() / rows_dropped.numel()
    p_nonzero_cols = torch.count_nonzero(
        cols_dropped).item() / cols_dropped.numel()

    assert abs(p_nonzero_rows - 0.8) < 0.1
    assert abs(p_nonzero_cols - 0.7) < 0.1

    assert torch.std(rows_dropped, dim=-2).sum() == 0
    assert torch.std(cols_dropped, dim=-3).sum() == 0
