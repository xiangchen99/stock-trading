#viewing account information
#get requests

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.requests import GetAccountRequest


trading_client = TradingClient('PK3K53ADW7Z1KLTP0MNG', 'pq3xJb53r0EtCaRY7ecpC4BzbElZEpqNAFeNEPP7')

# Get our account information.
account = trading_client.get_account()

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))

# Check our current balance vs. our balance at the last market close
balance_change = float(account.equity) - float(account.last_equity)
print(f'Today\'s portfolio balance change: ${balance_change}')