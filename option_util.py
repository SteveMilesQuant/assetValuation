from option_enum import OptionType
from model import ModelType, NumericalMethod
from option import Option
from gbm import euro_black_scholes_merton, gbm_binomial_tree, barrier_reiner_rubinstein
from pde import pde
from monte_carlo import monte_carlo


# To avoid circular imports, we have to make a separate function for the user
#   to add all evaluation methods to the mapping in the option object
def add_all_evaluation_methods(option: Option):
    evaluation_methods = {
        (ModelType.GBM, OptionType.EUROPEAN, NumericalMethod.CLOSED_FORM): euro_black_scholes_merton,
        (ModelType.GBM, OptionType.EUROPEAN, NumericalMethod.TREE): gbm_binomial_tree,
        (ModelType.GBM, OptionType.EUROPEAN, NumericalMethod.PDE): pde,
        (ModelType.GBM, OptionType.EUROPEAN, NumericalMethod.MONTE_CARLO): monte_carlo,
        
        (ModelType.GBM, OptionType.AMERICAN, NumericalMethod.CLOSED_FORM): None,
        (ModelType.GBM, OptionType.AMERICAN, NumericalMethod.TREE): gbm_binomial_tree,
        (ModelType.GBM, OptionType.AMERICAN, NumericalMethod.PDE): pde,
        (ModelType.GBM, OptionType.AMERICAN, NumericalMethod.MONTE_CARLO): monte_carlo,
        
        (ModelType.GBM, OptionType.BARRIER, NumericalMethod.CLOSED_FORM): barrier_reiner_rubinstein,
        (ModelType.GBM, OptionType.BARRIER, NumericalMethod.TREE): None,
        (ModelType.GBM, OptionType.BARRIER, NumericalMethod.PDE): pde,
        (ModelType.GBM, OptionType.BARRIER, NumericalMethod.MONTE_CARLO): monte_carlo
    }
    for eval_method_key, eval_method in evaluation_methods.items():
        option.add_evaluation_method(eval_method_key, eval_method)

