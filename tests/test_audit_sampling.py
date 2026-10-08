import pandas as pd
import pytest
from airvs.audit.sampling import assign_equal_size_strata, largest_remainder_allocation, stratified_sample

def test_assign_equal_size_strata_balanced_and_deterministic():
    df = pd.DataFrame({'molecule_id': [f'm{i}' for i in range(20)], 'canonical_smiles': [f'C{i}' for i in range(20)], 'score': [1.0] * 20})
    a = assign_equal_size_strata(df, 'score', n_strata=5)
    b = assign_equal_size_strata(df.sample(frac=1.0, random_state=7), 'score', n_strata=5)
    assert a.groupby('stratum').size().tolist() == [4, 4, 4, 4, 4]
    assert a[['molecule_id', 'stratum']].sort_values('molecule_id').reset_index(drop=True).equals(b[['molecule_id', 'stratum']].sort_values('molecule_id').reset_index(drop=True))

def test_largest_remainder_allocation_exact_total():
    alloc = largest_remainder_allocation({0: 10, 1: 20, 2: 30}, total=12, min_per_nonempty=2)
    assert sum(a.n_sample for a in alloc) == 12
    assert all(a.n_sample >= 2 for a in alloc)

def test_largest_remainder_rejects_too_small_total():
    with pytest.raises(ValueError):
        largest_remainder_allocation({0: 10, 1: 20, 2: 30}, total=5, min_per_nonempty=2)

def test_stratified_sample_no_duplicates():
    df = pd.DataFrame({'molecule_id': [f'm{i}' for i in range(50)], 'canonical_smiles': [f'C{i}' for i in range(50)], 'score': list(range(50))})
    df = assign_equal_size_strata(df, 'score', n_strata=5)
    sample = stratified_sample(df, q=10, seed=47001)
    assert len(sample) == 10
    assert not sample['molecule_id'].duplicated().any()
