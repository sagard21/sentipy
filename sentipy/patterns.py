##########################################
# RE patterns used during pre-processing #
# and cleaning data                      #
##########################################

#################
# Date patterns #
#################
# Seperators for date
seperators_ls = ['/', '-']
seperators_string = '|'.join(seperators_ls)

dd_mm_yy = r'\d\d\s*({})\s*\d\d({})\s*\d\d'.format(seperators_string,
                                                   seperators_string)
dd_mm_yyyy = r'\d\d\s*({})\s*\d\d({})\s*\d\d\d\d'.format(seperators_string,
                                                         seperators_string)

like_date = f'{dd_mm_yyyy}|{dd_mm_yy}'

####################
# Mentions pattern #
####################
like_mentions = r'@[a-z0-9_]+'

#################
# Link patterns #
#################
regular_links = r'http(s){0,1}://(\w|\d|-|_|\.)+'
short_links = r't.co(\w|\d|-|_|\.|/)+'

like_link = f'{regular_links}|{short_links}'

##################
# Number pattern #
##################
like_number = r'\d+'

####################
# Currency pattern #
####################
# Currencies list
currency_abbrs = ['USD', 'INR', 'JPY', 'GBP', 'CHF', 'EUR']
currency_symbols = [r'\$', '₹', '¥', '£', 'SFr.', '€']
all_currencies = currency_abbrs + currency_symbols
all_currencies_string = '|'.join(all_currencies)

like_currency = r'({})\s*\d+,*\d*\.*\d*'.format(all_currencies_string)
