#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
import pandas as pd
from examples.portfolio_optimization.data_request import PortfolioDataRequest
import requests
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import discrete_allocation
from pypfopt.discrete_allocation import DiscreteAllocation

class PortfolioOptimization:

    """
        Class for optimizing a historic portfolio
    """

    def __init__(self, table):
        mu = expected_returns.mean_historical_return(table)
        S = risk_models.sample_cov(table)

        # Optimise for maximal Sharpe ratio
        ef = EfficientFrontier(mu, S)
        ef.max_sharpe()  # Raw weights
        self.cleaned_weights = ef.clean_weights()
        print(self.cleaned_weights)
        ef.portfolio_performance(verbose=True)

        latest_prices = discrete_allocation.get_latest_prices(table)
        self.allocation, self.leftover = DiscreteAllocation(self.cleaned_weights, latest_prices, total_portfolio_value=10000).lp_portfolio()

    def report_discrete_allocation(self):
        print(self.allocation)
        print("Funds remaining: ${:.2f}".format(self.leftover))

    def get_cleaned_weights(self):
        return self.cleaned_weights


class PortfolioReturns:

    def __init__(self, stocks, discrete_allocation, start_date, end_date):
        data = PortfolioDataRequest(stocks, start_date, end_date).table
        self.start_date = start_date
        self.end_date = end_date
        starting_value = 0
        ending_value = 0
        for stock in stocks:
            try:
                # Initial value of portfolio
                starting_value += data[stock][0]*discrete_allocation[stock]
                # Ending portfolio value
                ending_value += \
                    data[stock][len(data[stock])-1]*discrete_allocation[stock]
            except KeyError:
                print(stock, ' received a weight of zero.')
                continue
        self.returns = [(ending_value - starting_value) / starting_value]

    def report_returns(self):
        print(
            'Portfolio Returns for ', self.start_date, ' to ', self.end_date,
            ' are ', self.returns
            )
