from airvs.data.registry import FOLDS, LIT_ASSAYS, SEEDS

def test_lit_registry_counts_and_main_tasks():
    assert len(LIT_ASSAYS) == 15
    assert sum(a.is_main for a in LIT_ASSAYS) == 11
    assert {a.name for a in LIT_ASSAYS if not a.is_main} == {'ESR_ago', 'ESR_antago', 'PPARG', 'TP53'}

def test_folds_cover_each_assay_once():
    folded = [x for xs in FOLDS.values() for x in xs]
    assert sorted(folded) == sorted(a.name for a in LIT_ASSAYS)

def test_seed_contract():
    assert SEEDS[0] == 47001
    assert SEEDS[-1] == 47020
    assert len(SEEDS) == 20
