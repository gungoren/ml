#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
import quandl
from examples.portfolio_optimization.quand_socket import QuandlSocket

class PortfolioDataRequest:

    """ stocks = [], start_date/end_date are strings 'YYYY-MM-DD' """
    """
        Class for portfolfio optimization, downloads relevant portfolio data
        and caches it for use in optimization without saving it locally.
    """

    def __init__(self, stocks, start_date, end_date):
        QuandlSocket()
        data = quandl.get_table(
            'WIKI/PRICES',
            ticker=stocks,
            qopts={'columns': ['date', 'ticker', 'adj_close']},
            date={'gte': start_date, 'lte': end_date},
            paginate=True
            )
        df = data.set_index('date')
        self.table = df.pivot(columns='ticker')
        # By specifying col[1] in below list comprehension
        # You can select the stock names under multi-level column
        self.table.columns = [col[1] for col in self.table.columns]