//+------------------------------------------------------------------+
//| MACrossoverTest.mq5 — Simple MA Crossover for Comparison        |
//|                                                                  |
//| Same logic as main.py's get_signal() for direct comparison.     |
//| Logs all trades to CSV for validation against Python backtester. |
//+------------------------------------------------------------------+
#property copyright "XAU Learning Model"
#property version   "1.00"
#property strict

//--- Input parameters
input double   LotSize        = 0.01;
input int      MagicNumber    = 888888;
input int      Slippage       = 30;
input int      MA_Fast_Period = 5;
input int      MA_Med_Period  = 10;
input int      MA_Cross_Period = 20;
input int      MA_Slow_Period = 100;
input string   LogFolder      = "tick_export";  // Same folder as tick data

//--- Handles
int handleMA5, handleMA10, handleMA20, handleMA100;

//--- Trade log
int logHandle = INVALID_HANDLE;
int tradeCount = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   handleMA5   = iMA(_Symbol, PERIOD_CURRENT, MA_Fast_Period, 0, MODE_SMA, PRICE_CLOSE);
   handleMA10  = iMA(_Symbol, PERIOD_CURRENT, MA_Med_Period, 0, MODE_SMA, PRICE_CLOSE);
   handleMA20  = iMA(_Symbol, PERIOD_CURRENT, MA_Cross_Period, 0, MODE_SMA, PRICE_CLOSE);
   handleMA100 = iMA(_Symbol, PERIOD_CURRENT, MA_Slow_Period, 0, MODE_SMA, PRICE_CLOSE);

   if(handleMA5 == INVALID_HANDLE || handleMA10 == INVALID_HANDLE ||
      handleMA20 == INVALID_HANDLE || handleMA100 == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create MA indicators");
      return(INIT_FAILED);
   }

   // Open trade log
   string logFile = LogFolder + "\\mt5_trades.csv";
   logHandle = FileOpen(logFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(logHandle != INVALID_HANDLE)
   {
      FileWrite(logHandle, "ticket", "type", "entry_price", "exit_price",
                "lots", "profit", "commission", "swap", "open_time", "close_time",
                "comment");
   }

   Print("MACrossoverTest initialized for comparison testing");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Log all closed trades from history
   ExportTradeHistory();

   if(logHandle != INVALID_HANDLE)
   {
      FileClose(logHandle);
      Print("Trade log saved. Total trades: ", tradeCount);
   }

   IndicatorRelease(handleMA5);
   IndicatorRelease(handleMA10);
   IndicatorRelease(handleMA20);
   IndicatorRelease(handleMA100);
}

//+------------------------------------------------------------------+
void OnTick()
{
   // Only process on new bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBar == lastBar) return;
   lastBar = currentBar;

   // Get MA values from completed bar
   double ma5[1], ma10[1], ma20[1], ma100[1];
   if(CopyBuffer(handleMA5, 0, 1, 1, ma5) != 1) return;
   if(CopyBuffer(handleMA10, 0, 1, 1, ma10) != 1) return;
   if(CopyBuffer(handleMA20, 0, 1, 1, ma20) != 1) return;
   if(CopyBuffer(handleMA100, 0, 1, 1, ma100) != 1) return;

   bool hasPosition = false;
   int posType = 0;

   // Check current position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      hasPosition = true;
      posType = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? 1 : -1;
      break;
   }

   // Signal logic — exact match to Python's get_signal()
   string signal = "HOLD";

   if(hasPosition)
   {
      if(posType == 1 && ma5[0] < ma20[0])   signal = "CLOSE";
      if(posType == -1 && ma5[0] > ma20[0])   signal = "CLOSE";
   }
   else
   {
      if(ma5[0] > ma20[0] && ma10[0] > ma100[0]) signal = "BUY";
      if(ma5[0] < ma20[0] && ma10[0] < ma100[0]) signal = "SELL";
   }

   // Execute
   if(signal == "BUY")    ExecuteBuy();
   if(signal == "SELL")   ExecuteSell();
   if(signal == "CLOSE")  CloseAll();
}

//+------------------------------------------------------------------+
void ExecuteBuy()
{
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = _Symbol;
   request.volume    = LotSize;
   request.type      = ORDER_TYPE_BUY;
   request.price     = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.deviation = Slippage;
   request.magic     = MagicNumber;
   request.comment   = "MA_BUY";

   long fillType = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((fillType & SYMBOL_FILLING_IOC) != 0) request.type_filling = ORDER_FILLING_IOC;
   else if((fillType & SYMBOL_FILLING_FOK) != 0) request.type_filling = ORDER_FILLING_FOK;
   else request.type_filling = ORDER_FILLING_RETURN;

   OrderSend(request, result);
}

//+------------------------------------------------------------------+
void ExecuteSell()
{
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   request.action    = TRADE_ACTION_DEAL;
   request.symbol    = _Symbol;
   request.volume    = LotSize;
   request.type      = ORDER_TYPE_SELL;
   request.price     = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   request.deviation = Slippage;
   request.magic     = MagicNumber;
   request.comment   = "MA_SELL";

   long fillType = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((fillType & SYMBOL_FILLING_IOC) != 0) request.type_filling = ORDER_FILLING_IOC;
   else if((fillType & SYMBOL_FILLING_FOK) != 0) request.type_filling = ORDER_FILLING_FOK;
   else request.type_filling = ORDER_FILLING_RETURN;

   OrderSend(request, result);
}

//+------------------------------------------------------------------+
void CloseAll()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      MqlTradeRequest request = {};
      MqlTradeResult  result  = {};
      request.action    = TRADE_ACTION_DEAL;
      request.symbol    = _Symbol;
      request.volume    = volume;
      request.position  = ticket;
      request.deviation = Slippage;
      request.magic     = MagicNumber;
      request.comment   = "MA_CLOSE";

      if(type == POSITION_TYPE_BUY)
      {
         request.type  = ORDER_TYPE_SELL;
         request.price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      }
      else
      {
         request.type  = ORDER_TYPE_BUY;
         request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      }

      long fillType = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
      if((fillType & SYMBOL_FILLING_IOC) != 0) request.type_filling = ORDER_FILLING_IOC;
      else if((fillType & SYMBOL_FILLING_FOK) != 0) request.type_filling = ORDER_FILLING_FOK;
      else request.type_filling = ORDER_FILLING_RETURN;

      OrderSend(request, result);
   }
}

//+------------------------------------------------------------------+
void ExportTradeHistory()
{
   if(logHandle == INVALID_HANDLE) return;

   // Select all history
   HistorySelect(0, TimeCurrent());

   int total = HistoryDealsTotal();
   Print("Exporting ", total, " history deals...");

   for(int i = 0; i < total; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;
      if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != MagicNumber) continue;

      long dealType  = HistoryDealGetInteger(ticket, DEAL_TYPE);
      long dealEntry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      double price   = HistoryDealGetDouble(ticket, DEAL_PRICE);
      double volume  = HistoryDealGetDouble(ticket, DEAL_VOLUME);
      double profit  = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      double commission = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      double swap    = HistoryDealGetDouble(ticket, DEAL_SWAP);
      datetime time  = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      string comment = HistoryDealGetString(ticket, DEAL_COMMENT);

      string typeStr = (dealType == DEAL_TYPE_BUY) ? "BUY" : "SELL";
      string entryStr = "";
      if(dealEntry == DEAL_ENTRY_IN)  entryStr = "IN";
      if(dealEntry == DEAL_ENTRY_OUT) entryStr = "OUT";

      FileWrite(logHandle,
                IntegerToString((long)ticket),
                typeStr + "_" + entryStr,
                DoubleToString(price, _Digits),
                "",  // exit_price filled by matching
                DoubleToString(volume, 2),
                DoubleToString(profit, 2),
                DoubleToString(commission, 2),
                DoubleToString(swap, 2),
                TimeToString(time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
                "",  // close_time
                comment);

      tradeCount++;
   }
}
//+------------------------------------------------------------------+
