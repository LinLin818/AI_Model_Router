#Configuration

#List the models from cheapest to the priciest
TIERS = [{},{},{}]

# which model is currently answering the prompt
JUDGE_TIER = 0

#The most amount of time a request can move up
MAXEST_ESCALATIONS = 2

# An answer that needs to be accepted without escalating
JUDGE_MIN_SCORE = 4

#The file where each request's stats get appended
LOG_PATH = "router_log.json"


