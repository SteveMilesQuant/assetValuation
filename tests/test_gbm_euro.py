import pytest
from gbm.euro import euro_put, euro_put_binomial

def test_gbm_euro():
    # From Haug p3
    test_Haug_p3 = euro_put(60, 65, 0.08, 0, 0.2, 0.25)
    assert test_Haug_p3 == 4.754368137482118

    test_closedform = euro_put(60, 65, 0.08, 0.01, 0.2, 0.25)
    test_binomial = euro_put_binomial(60, 65, 0.08, 0.01, 0.2, 0.25, 1000)
    relDiff = abs(test_binomial-test_closedform)/test_closedform
    assert relDiff < 1e-4



