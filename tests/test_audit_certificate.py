from airvs.audit.certificate import StratumAudit, hypergeom_lower_bound, stratified_lower_bound

def test_hypergeom_lower_bound_edges():
    assert hypergeom_lower_bound(0, 0, 0) == 0
    assert hypergeom_lower_bound(100, 10, 0) == 0
    assert hypergeom_lower_bound(10, 10, 4) == 4

def test_hypergeom_lower_bound_monotone_in_k():
    vals = [hypergeom_lower_bound(1000, 50, k, alpha=0.05) for k in range(6)]
    assert vals == sorted(vals)

def test_stratified_lower_bound_sums_local_bounds():
    total, bounds = stratified_lower_bound([StratumAudit(100, 10, 1), StratumAudit(200, 20, 2)])
    assert total == sum(b.lower_active for b in bounds)
    assert len(bounds) == 2
