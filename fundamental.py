from flask import Flask, jsonify
import pandas as pd
import yfinance as yf
import numpy as np

def get_fundamental_details(ticker):
    # Fetch the stock data using yfinance
    stock = yf.Ticker(ticker)

    # Fetch fundamental data and financial statements
    fundamental_data = stock.info   
    income_statement = stock.financials
    balance_sheet = stock.balance_sheet
    cash_flow_statement = stock.cashflow

   
    current_net_income = income_statement.loc['Net Income'].iloc[0]  
    previous_net_income = income_statement.loc['Net Income'].iloc[1] 

    earnings_growth_yoy = 'N/A'
    # Calculate Earnings Growth (YoY)
    if current_net_income is not None and previous_net_income is not None:
        earnings_growth_yoy = ((current_net_income - previous_net_income) / previous_net_income) * 100       
     
    current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else None
    current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else None

    current_ratio = 'N/A'
    # Calculate Earnings Growth (YoY)
    if current_assets is not None and current_liabilities is not None:
       current_ratio =  current_assets/current_liabilities   

    # Ensure data is transposed for proper indexing
    balance_sheet = balance_sheet.T

    # Get the latest non-null Debt-to-Equity Ratio
    debt_to_equity = "N/A"
    total_liabilities_series = total_equity_series = None

    if "Total Liabilities Net Minority Interest" in balance_sheet.columns and "Ordinary Shares Number" in balance_sheet.columns:
        total_liabilities_series = balance_sheet["Total Liabilities Net Minority Interest"].dropna()
        total_equity_series = balance_sheet["Ordinary Shares Number"].dropna()

        if not total_liabilities_series.empty and not total_equity_series.empty:
            total_liabilities = total_liabilities_series.iloc[0]  # Get the latest non-null value
            total_equity = total_equity_series.iloc[0]  # Get the latest non-null value

            if total_liabilities is not None and total_equity is not None and total_equity != 0:
                debt_to_equity = total_liabilities / total_equity




    fundamental_metrics = {
    'EPS': fundamental_data.get('epsTrailingTwelveMonths', None),
    'P/E Ratio': fundamental_data.get('trailingPE', None),
    'Revenue Growth': fundamental_data.get('revenueGrowth', None),
    'Debt-to-Equity Ratio': debt_to_equity,  
    'Earnings Growth(YoY)': earnings_growth_yoy             
   
}
    
    eps = fundamental_metrics['EPS']
    pe_ratio = fundamental_metrics['P/E Ratio']
    revenue_growth = fundamental_metrics['Revenue Growth']
    debt_to_equity = fundamental_metrics['Debt-to-Equity Ratio']
    earnings_growth_yoy = fundamental_metrics['Earnings Growth(YoY)']
  
      
    
    # Initialize status variables
    earnings_growth_status = "none"
    debt_to_equity_status = "none"
    pe_ratio_status = "none"
    revenue_growth_status = "none"   
    eps_status = "none"

      # Updated Thresholds
    earnings_growth_threshold = 0.12      # Earnings Growth > 12% (good)
    debt_to_equity_threshold = 1.5        # D/E < 1.5 (good)
    pe_ratio_threshold = 25               # P/E < 25 (good)
    revenue_growth_threshold = 0.12       # Revenue Growth > 12% (good)
    eps_threshold = 0                     # EPS > 0 (good)

    # Updated Weightages
    earnings_growth_weight = 2       # 3.5% weight for earnings growth
    debt_to_equity_weight = 1.5          # 2.75% weight for debt-to-equity ratio
    pe_ratio_weight = 1.5                 # 2.5% weight for P/E ratio
    revenue_growth_weight = 2         # 3.5% weight for revenue growth
    eps_weight = 3                   # 3.75% weight for EPS

    # Check and assign status for EPS, if available
    if eps is not None:
        if eps > eps_threshold:
            eps_status = "good"
        else:
            eps_status = "bad"

    # Check and assign status for earnings growth, if available
    if earnings_growth_yoy is not None:
        if earnings_growth_yoy >= earnings_growth_threshold:
            earnings_growth_status = "good"
        else:
            earnings_growth_status = "bad"

    # Check and assign status for debt-to-equity, if available
    if debt_to_equity is not None:
        if debt_to_equity <= debt_to_equity_threshold:
            debt_to_equity_status = "good"
        else:
            debt_to_equity_status = "bad"

    # Check and assign status for P/E ratio, if available
    if pe_ratio is not None:
        if pe_ratio <= pe_ratio_threshold:
            pe_ratio_status = "good"
        else:
            pe_ratio_status = "bad"

    # Check and assign status for revenue growth, if available
    if revenue_growth is not None:
        if revenue_growth >= revenue_growth_threshold:
            revenue_growth_status = "good"
        else:
            revenue_growth_status = "bad"

    

    # Calculate overall score
    overall_fa_score = 0
    overall_fa_score += eps_weight if eps_status == "good" else 0
    overall_fa_score += earnings_growth_weight if earnings_growth_status == "good" else 0
    overall_fa_score += debt_to_equity_weight if debt_to_equity_status == "good" else 0
    overall_fa_score += pe_ratio_weight if pe_ratio_status == "good" else 0
    overall_fa_score += revenue_growth_weight if revenue_growth_status == "good" else 0
  

    earnings_growth = stock.info.get('earningsGrowth', None)

    peg_ratio = None
    if pe_ratio is not None and earnings_growth is not None and earnings_growth != 0:
        peg_ratio = pe_ratio / earnings_growth


    # Financial Metrics (Valuation, Profitability, etc.)
    financialMetrics = {
        'Market Cap': fundamental_data.get('marketCap', None),       
        'Price-to-Book (P/B) Ratio': fundamental_data.get('priceToBook', None),
        'Price-to-Sales (P/S) Ratio': fundamental_data.get('priceToSalesTrailing12Months', None),
        'PEG Ratio': peg_ratio,
        'EV/EBITDA': fundamental_data.get('enterpriseToEbitda', None)                  
    }

    # Balance Sheet Data
    balanceSheetInformation = {
        'Total Assets': balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else None,
        'Total Liabilities': balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None,
        'Total Stockholder Equity': (balance_sheet.loc['Total Assets'].iloc[0] - balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]) 
        if 'Total Assets' in balance_sheet.index and 'Total Liabilities Net Minority Interest' in balance_sheet.index else None,
        'Long Term Debt': balance_sheet.loc['Long Term Debt'].iloc[0] if 'Long Term Debt' in balance_sheet.index else None,
        'Current Assets': balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else None,
        'Current Liabilities': balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else None,
        'Inventory': balance_sheet.loc['Inventory'].iloc[0] if 'Inventory' in balance_sheet.index else None       
    }


    # Income Statement Data
    incomeStatement = {
        'Total Revenue': income_statement.loc['Total Revenue'].iloc[0] if 'Total Revenue' in income_statement.index else None,
        'Operating Income': income_statement.loc['Operating Income'].iloc[0] if 'Operating Income' in income_statement.index else None,
        'Net Income': income_statement.loc['Net Income'].iloc[0] if 'Net Income' in income_statement.index else None,
        'Gross Profit': income_statement.loc['Gross Profit'].iloc[0] if 'Gross Profit' in income_statement.index else None             
    }

    # Growth Indicators
    growthIndicators = {
         'Revenue Growth (YoY)': income_statement.loc['Revenue Growth'].iloc[0] if 'Revenue Growth' in income_statement.index else None,        
        'Profit Margins': (incomeStatement['Net Income'] / incomeStatement['Total Revenue']) * 100 if incomeStatement['Total Revenue'] else None,
        'ROE (Return on Equity)': balanceSheetInformation.get('Return on Equity (ROE)', None),
        'ROA (Return on Assets)': balanceSheetInformation.get('Return on Assets (ROA)', None)
    }

    # Assuming your original cashFlowStatement dict is defined here
    cashFlowStatement = {
        'Operating Cash Flow': cash_flow_statement.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cash_flow_statement.index else None,
        'Investing Cash Flow': cash_flow_statement.loc['Investing Cash Flow'].iloc[0] if 'Investing Cash Flow' in cash_flow_statement.index else None,
        'Financing Cash Flow': cash_flow_statement.loc['Financing Cash Flow'].iloc[0] if 'Financing Cash Flow' in cash_flow_statement.index else None,
        'Cash Flow to Debt Ratio': (
            cash_flow_statement.loc['Operating Cash Flow'].iloc[0] / balance_sheet.loc['Long Term Debt'].iloc[0]
            if 'Operating Cash Flow' in cash_flow_statement.index and 'Long Term Debt' in balance_sheet.index
            else None
        )
    }

    # Replace NaN or None values with 'N/A'
    cashFlowStatement = {k: ('N/A' if pd.isna(v) else v) for k, v in cashFlowStatement.items()}

    # Company Overview Data
    companyOverview = {
        'Company Name': fundamental_data.get('longName', None),
        'Sector': fundamental_data.get('sector', None),
        'Industry': fundamental_data.get('industry', None)
    }

    # Risk Indicators
    riskIndicators = {
        'Debt-to-Equity Ratio(Risk)': balanceSheetInformation.get('Long Term Debt', None) / balanceSheetInformation.get('Total Stockholder Equity', None) 
        if balanceSheetInformation.get('Long Term Debt') and balanceSheetInformation.get('Total Stockholder Equity') else None,
        'Interest Coverage Ratio': income_statement.loc['Interest Coverage'].iloc[0] if 'Interest Coverage' in income_statement.index else None,
        'Beta (Stock Volatility)': fundamental_data.get('beta', None),       
        'Quick Ratio': (balanceSheetInformation.get('Current Assets', None) - balanceSheetInformation.get('Inventory', None)) / balanceSheetInformation.get('Current Liabilities', None)
        if balanceSheetInformation.get('Current Assets') and balanceSheetInformation.get('Inventory') and balanceSheetInformation.get('Current Liabilities') else None
    }

    # Dividends
    dividends = {
        'Payout Ratio': fundamental_data.get('payoutRatio', None),
        'Dividend Growth Rate': fundamental_data.get('dividendGrowthRate', None)
    }

     # Profitability Indicators
    profitabilityIndicators = {
        'Gross Margin': (income_statement.loc['Gross Profit'].iloc[0] / income_statement.loc['Total Revenue'].iloc[0]) * 100 
                         if 'Gross Profit' in income_statement.index and 'Total Revenue' in income_statement.index else None,
        'Operating Margin': (income_statement.loc['Operating Income'].iloc[0] / income_statement.loc['Total Revenue'].iloc[0]) * 100 
                            if 'Operating Income' in income_statement.index and 'Total Revenue' in income_statement.index else None,
        'Net Margin': (income_statement.loc['Net Income'].iloc[0] / income_statement.loc['Total Revenue'].iloc[0]) * 100 
                      if 'Net Income' in income_statement.index and 'Total Revenue' in income_statement.index else None
    }

    # Liquidity Indicators
    liquidityIndicators = {
        'Cash Ratio': balance_sheet.loc['Cash And Cash Equivalents'].iloc[0] / balance_sheet.loc['Current Liabilities'].iloc[0] 
                      if 'Cash And Cash Equivalents' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index else None,
        'Working Capital': balance_sheet.loc['Current Assets'].iloc[0] - balance_sheet.loc['Current Liabilities'].iloc[0] 
                           if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index else None
    }

    investorInsightMetrics = {                
     'EPS': eps,
     'P/E Ratio': pe_ratio,
     'Revenue Growth': revenue_growth,
     'Debt-to-Equity Ratio': debt_to_equity,  
     'Earnings Growth(YoY)': earnings_growth_yoy            
           }

    result = {   

     'EPS': eps_status,
     'P/E Ratio': pe_ratio_status,
     'Revenue Growth': revenue_growth_status,
     'Debt-to-Equity Ratio': debt_to_equity_status,  
     'Earnings Growth(YoY)': earnings_growth_status 
    }

    def replace_nan_with_na(data):
        if isinstance(data, dict):
            return {k: replace_nan_with_na(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [replace_nan_with_na(v) for v in data]
        elif pd.isna(data):
            return 'N/A'
        else:
            return data




    return replace_nan_with_na({
    "fa_strategy": result,
    "overall_fa_score": round(overall_fa_score, 2),
    "Investor Insight Metrics": investorInsightMetrics,
    "Company Overview": companyOverview,
    "Growth Indicators": growthIndicators,
    "Risk Indicators": riskIndicators,
    "Dividends": dividends,
    "Cash Flow Statement": cashFlowStatement,
    "Financial Metrics": financialMetrics,
    "Income Statement": incomeStatement,
    "BalanceSheet Information": balanceSheetInformation,
    "Profitability Indicators": profitabilityIndicators,
    "Liquidity Indicators": liquidityIndicators,
    })



    





