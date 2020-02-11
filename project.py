import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.factors import SimpleMovingAverage
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.data.psychsignal import stocktwits
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.data.sentdex import sentiment #Sentdex News Sentiment
from quantopian.pipeline.data.psychsignal import twitter_withretweets as twitter_sentiment #PsychSignal Trader Mood


MAX_GROSS_LEVERAGE = 1.0
TOTAL_POSITIONS = 600


MAX_SHORT_POSITION_SIZE = 2.0 / TOTAL_POSITIONS 
MAX_LONG_POSITION_SIZE = 2.0 / TOTAL_POSITIONS 


def initialize(context):

    algo.attach_pipeline(make_pipeline(), 'long_short_equity_template')

    algo.attach_pipeline(risk_loading_pipeline(), 'risk_factors')

    algo.schedule_function(func=rebalance,
                           date_rule=algo.date_rules.week_start(),
                           time_rule=algo.time_rules.market_open(hours=0, minutes=30),
                           half_days=True)

    algo.schedule_function(func=record_vars,
                           date_rule=algo.date_rules.every_day(),
                           time_rule=algo.time_rules.market_close(),
                           half_days=True)


def make_pipeline():


    value = Fundamentals.ebit.latest / Fundamentals.enterprise_value.latest
    quality = Fundamentals.roe.latest

    sentiment_score = SimpleMovingAverage(inputs=[stocktwits.bull_minus_bear],window_length=3,) #Sentdex News Sentiment score
    mean_sentiment_5day = SimpleMovingAverage(inputs=[sentiment.sentiment_signal], window_length=5) #PsychSignal Trader Mood score
    psychSignal_score = SimpleMovingAverage(inputs=[twitter_sentiment.bull_bear_msg_ratio],window_length=5,)

    total_revenue = Fundamentals.total_revenue.latest
    capital_stock = Fundamentals.capital_stock.latest
    

    universe = QTradableStocksUS()

    # We winsorize our factor values in order to lessen the impact of outliers
    
    value_winsorized = value.winsorize(min_percentile=0.20, max_percentile=0.80)
    quality_winsorized = quality.winsorize(min_percentile=0.20, max_percentile=0.80)
    sentiment_score_winsorized = sentiment_score.winsorize(min_percentile=0.20, max_percentile=0.80)
    mean_sentiment_5day_winsorized = mean_sentiment_5day.winsorize(min_percentile=0.20, max_percentile=0.80)
    psychSignal_score_winsorized = psychSignal_score.winsorize(min_percentile=0.20, max_percentile=0.80)
    total_revenue_winsorized = total_revenue.winsorize(min_percentile=0.20, max_percentile=0.80)
    capital_stock_winsorized = capital_stock.winsorize(min_percentile=0.20, max_percentile=0.80)

    
    
    # Here we combine our winsorized factors, z-scoring them to equalize their influence
    combined_factor = (
        value_winsorized.zscore() +
        8*quality_winsorized.zscore()+
        sentiment_score_winsorized.zscore()+
        mean_sentiment_5day_winsorized.zscore()+
        4*psychSignal_score_winsorized.zscore()+
        5*total_revenue_winsorized.zscore()+
        10*capital_stock_winsorized.zscore()
        
    )

    longs = combined_factor.top(TOTAL_POSITIONS//2, mask=universe)
    shorts = combined_factor.bottom(TOTAL_POSITIONS//2, mask=universe)

    long_short_screen = (longs | shorts)

    # Create pipeline
    pipe = Pipeline(
        columns={
            'longs': longs,
            'shorts': shorts,
            'combined_factor': combined_factor
        },
        screen=long_short_screen
    )
    return pipe


def before_trading_start(context, data):
    # Call algo.pipeline_output to get the output
    # Note: this is a dataframe where the index is the SIDs for all
    # securities to pass my screen and the columns are the factors
    # added to the pipeline object above
    context.pipeline_data = algo.pipeline_output('long_short_equity_template')

    # This dataframe will contain all of our risk loadings
    context.risk_loadings = algo.pipeline_output('risk_factors')


def record_vars(context, data):

    # Plot the number of positions over time.
    algo.record(num_positions=len(context.portfolio.positions))


# Called at the start of every month in order to rebalance
# the longs and shorts lists
def rebalance(context, data):
    # Retrieve pipeline output
    pipeline_data = context.pipeline_data

    risk_loadings = context.risk_loadings

    objective = opt.MaximizeAlpha(pipeline_data.combined_factor)

    # Define the list of constraints
    constraints = []
    # Constrain our maximum gross leverage
    constraints.append(opt.MaxGrossExposure(MAX_GROSS_LEVERAGE))

    # Require our algorithm to remain dollar neutral
    constraints.append(opt.DollarNeutral())

    # Add the RiskModelExposure constraint to make use of the
    # default risk model constraints
    neutralize_risk_factors = opt.experimental.RiskModelExposure(
        risk_model_loadings=risk_loadings,
        version=0
    )
    constraints.append(neutralize_risk_factors)

    # With this constraint we enforce that no position can make up
    # greater than MAX_SHORT_POSITION_SIZE on the short side and
    # no greater than MAX_LONG_POSITION_SIZE on the long side. This
    # ensures that we do not overly concentrate our portfolio in
    # one security or a small subset of securities.
    constraints.append(
        opt.PositionConcentration.with_equal_bounds(
            min=-MAX_SHORT_POSITION_SIZE,
            max=MAX_LONG_POSITION_SIZE
        ))

    # Put together all the pieces we defined above by passing
    # them into the algo.order_optimal_portfolio function. This handles
    # all of our ordering logic, assigning appropriate weights
    # to the securities in our universe to maximize our alpha with
    # respect to the given constraints.
    algo.order_optimal_portfolio(
        objective=objective,
        constraints=constraints
    )