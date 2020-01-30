#
#
# Mehmet Güngören (mehmetgungoren@lyrebirdstudio.net)
#
#
import quandl

API_KEY = "jQLyPAzMoraqRrsj4V_j"

class QuandlSocket:

    """
        Socket for cached historical market data requests
    """

    def __init__(self):
        quandl.ApiConfig.api_key = API_KEY