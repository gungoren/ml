#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#

from examples.portfolio_optimization.port_opt import *

"""
    Script for portfolio optimization pipeline research
"""

if __name__ == '__main__':
    stocks = 'AAPL MSFT JNJ JPM XOM WMT UNH PFE VZ V BA'.split()
    data = PortfolioDataRequest(
            stocks,
            '2010-01-01',
            '2017-01-01'
            )
    optimization = PortfolioOptimization(data.table)
    optimization.report_discrete_allocation()
    returns = PortfolioReturns(
            stocks,
            optimization.allocation,
            '2017-01-01',
            '2018-01-01'
            )
    returns.report_returns()