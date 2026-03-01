//+------------------------------------------------------------------+
//| RLBridge.mq5 — RL Model Bridge Expert Advisor                    |
//|                                                                  |
//| Runs in Strategy Tester or Live. Communicates with Python RL     |
//| model via file-based messaging in Common\Files\rl_bridge\.       |
//|                                                                  |
//| Protocol:                                                        |
//|   1. EA writes state to state.csv                                |
//|   2. Python reads state, writes action to action.csv             |
//|   3. EA reads action, executes order, writes result to result.csv|
//|   4. Python reads result (reward/done), repeats                  |
//+------------------------------------------------------------------+
#property copyright "XAU Learning Model"
#property link      ""
#property version   "1.00"
#property strict

//--- Input parameters
input double   LotSize        = 0.01;       // Trade lot size
input int      Slippage       = 30;         // Allowed slippage in points
input int      MagicNumber    = 777777;     // EA magic number
input string   BridgeFolder   = "rl_bridge"; // Subfolder in Common\Files
input int      TimeoutMs      = 5000;       // Max wait for Python response (ms)
input int      PollIntervalMs = 1;          // Poll interval for file check (ms)

//--- File paths (set in OnInit)
string stateFile;
string actionFile;
string resultFile;
string resetFile;
string readyFile;

//--- State tracking
int    currentPosition = 0;   // 0=flat, 1=long, -1=short
double entryPrice      = 0.0;
ulong  positionTicket  = 0;
int    barCount        = 0;
bool   episodeDone     = false;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // Set file paths — using Common folder so Python can access them
   stateFile  = BridgeFolder + "\\state.csv";
   actionFile = BridgeFolder + "\\action.csv";
   resultFile = BridgeFolder + "\\result.csv";
   resetFile  = BridgeFolder + "\\reset.csv";
   readyFile  = BridgeFolder + "\\ready.csv";

   // Clean up old files
   FileDelete(stateFile, FILE_COMMON);
   FileDelete(actionFile, FILE_COMMON);
   FileDelete(resultFile, FILE_COMMON);
   FileDelete(resetFile, FILE_COMMON);

   // Write ready signal
   int h = FileOpen(readyFile, FILE_WRITE | FILE_CSV | FILE_COMMON);
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "ready", TimeToString(TimeCurrent()));
      FileClose(h);
   }

   Print("RLBridge EA initialized. Bridge folder: Common\\Files\\", BridgeFolder);
   Print("Lot: ", LotSize, " | Magic: ", MagicNumber, " | Timeout: ", TimeoutMs, "ms");

   currentPosition = 0;
   entryPrice = 0.0;
   barCount = 0;
   episodeDone = false;

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Write final done state
   WriteResult(0, 0.0, 0.0, AccountInfoDouble(ACCOUNT_BALANCE),
               AccountInfoDouble(ACCOUNT_EQUITY), currentPosition, true);

   // Clean up
   FileDelete(readyFile, FILE_COMMON);
   Print("RLBridge EA deinitialized. Total bars processed: ", barCount);
}

//+------------------------------------------------------------------+
//| Expert tick function — called on every new tick/bar               |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only process on new bar
   static datetime lastBar = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(currentBar == lastBar) return;
   lastBar = currentBar;

   if(episodeDone) return;

   barCount++;

   // Check for reset signal from Python
   if(FileIsExist(resetFile, FILE_COMMON))
   {
      HandleReset();
      return;
   }

   // Step 1: Gather market state
   double open    = iOpen(_Symbol, PERIOD_CURRENT, 1);
   double high    = iHigh(_Symbol, PERIOD_CURRENT, 1);
   double low     = iLow(_Symbol, PERIOD_CURRENT, 1);
   double close   = iClose(_Symbol, PERIOD_CURRENT, 1);
   long   volume  = iVolume(_Symbol, PERIOD_CURRENT, 1);
   double bid     = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask     = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double spread  = (ask - bid) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   // Account state
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);

   // Position state
   UpdatePositionState();
   double unrealizedPnl = GetUnrealizedPnl();

   // Step 2: Write state file
   WriteState(TimeCurrent(), open, high, low, close, volume,
              bid, ask, spread, currentPosition, entryPrice,
              unrealizedPnl, balance, equity);

   // Step 3: Wait for Python to write action
   int action = WaitForAction();
   if(action < 0)
   {
      Print("ERROR: Timeout waiting for Python action at bar ", barCount);
      return;
   }

   // Step 4: Execute action
   double realizedPnl = 0.0;
   double fillPrice = 0.0;
   int executedAction = action;

   if(action == 1 && currentPosition == 0)       // BUY
   {
      fillPrice = ExecuteBuy();
      if(fillPrice > 0) currentPosition = 1;
      else executedAction = 0; // Failed, treated as HOLD
   }
   else if(action == 2 && currentPosition == 0)   // SELL
   {
      fillPrice = ExecuteSell();
      if(fillPrice > 0) currentPosition = -1;
      else executedAction = 0;
   }
   else if(action == 3 && currentPosition != 0)   // CLOSE
   {
      realizedPnl = ClosePosition();
      fillPrice = (currentPosition == 1) ? bid : ask;
      currentPosition = 0;
      entryPrice = 0.0;
   }
   else
   {
      executedAction = 0; // HOLD or invalid
   }

   // Update balance/equity after execution
   balance = AccountInfoDouble(ACCOUNT_BALANCE);
   equity  = AccountInfoDouble(ACCOUNT_EQUITY);

   // Step 5: Write result
   WriteResult(executedAction, fillPrice, realizedPnl,
               balance, equity, currentPosition, false);
}

//+------------------------------------------------------------------+
//| Write market state to file                                        |
//+------------------------------------------------------------------+
void WriteState(datetime time, double open, double high, double low,
                double close, long volume, double bid, double ask,
                double spread, int position, double entry,
                double unrealizedPnl, double balance, double equity)
{
   // Delete old file first
   FileDelete(stateFile, FILE_COMMON);

   int handle = FileOpen(stateFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot write state file. Error: ", GetLastError());
      return;
   }

   // Header
   FileWrite(handle, "timestamp", "open", "high", "low", "close", "volume",
             "bid", "ask", "spread", "position", "entry_price",
             "unrealized_pnl", "balance", "equity", "bar_count");

   // Data
   FileWrite(handle, TimeToString(time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
             DoubleToString(open, _Digits),
             DoubleToString(high, _Digits),
             DoubleToString(low, _Digits),
             DoubleToString(close, _Digits),
             IntegerToString(volume),
             DoubleToString(bid, _Digits),
             DoubleToString(ask, _Digits),
             DoubleToString(spread, 1),
             IntegerToString(position),
             DoubleToString(entry, _Digits),
             DoubleToString(unrealizedPnl, 2),
             DoubleToString(balance, 2),
             DoubleToString(equity, 2),
             IntegerToString(barCount));

   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Wait for Python to write action file                              |
//+------------------------------------------------------------------+
int WaitForAction()
{
   // Delete old action file if exists
   if(FileIsExist(actionFile, FILE_COMMON))
      FileDelete(actionFile, FILE_COMMON);

   // Wait with timeout
   uint startTime = GetTickCount();

   while(GetTickCount() - startTime < (uint)TimeoutMs)
   {
      if(FileIsExist(actionFile, FILE_COMMON))
      {
         // Small delay to ensure file is fully written
         Sleep(PollIntervalMs);

         int handle = FileOpen(actionFile, FILE_READ | FILE_CSV | FILE_COMMON, ',');
         if(handle != INVALID_HANDLE)
         {
            int action = (int)FileReadNumber(handle);
            FileClose(handle);
            FileDelete(actionFile, FILE_COMMON);
            return action;
         }
      }
      Sleep(PollIntervalMs);
   }

   return -1; // Timeout
}

//+------------------------------------------------------------------+
//| Write execution result to file                                    |
//+------------------------------------------------------------------+
void WriteResult(int executedAction, double fillPrice, double realizedPnl,
                 double balance, double equity, int position, bool done)
{
   FileDelete(resultFile, FILE_COMMON);

   int handle = FileOpen(resultFile, FILE_WRITE | FILE_CSV | FILE_COMMON, ',');
   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot write result file. Error: ", GetLastError());
      return;
   }

   FileWrite(handle, "action", "fill_price", "realized_pnl",
             "balance", "equity", "position", "done");

   FileWrite(handle, IntegerToString(executedAction),
             DoubleToString(fillPrice, _Digits),
             DoubleToString(realizedPnl, 2),
             DoubleToString(balance, 2),
             DoubleToString(equity, 2),
             IntegerToString(position),
             done ? "1" : "0");

   FileClose(handle);
}

//+------------------------------------------------------------------+
//| Execute a BUY market order                                        |
//+------------------------------------------------------------------+
double ExecuteBuy()
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
   request.comment   = "RLBridge BUY";

   // Determine filling mode
   long fillType = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((fillType & SYMBOL_FILLING_IOC) != 0)
      request.type_filling = ORDER_FILLING_IOC;
   else if((fillType & SYMBOL_FILLING_FOK) != 0)
      request.type_filling = ORDER_FILLING_FOK;
   else
      request.type_filling = ORDER_FILLING_RETURN;

   if(!OrderSend(request, result))
   {
      Print("BUY failed: ", result.retcode, " - ", result.comment);
      return 0.0;
   }

   if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
   {
      entryPrice = result.price;
      positionTicket = result.deal;
      Print("BUY executed @ ", result.price, " | Ticket: ", result.deal);
      return result.price;
   }

   Print("BUY retcode: ", result.retcode);
   return 0.0;
}

//+------------------------------------------------------------------+
//| Execute a SELL market order                                       |
//+------------------------------------------------------------------+
double ExecuteSell()
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
   request.comment   = "RLBridge SELL";

   long fillType = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   if((fillType & SYMBOL_FILLING_IOC) != 0)
      request.type_filling = ORDER_FILLING_IOC;
   else if((fillType & SYMBOL_FILLING_FOK) != 0)
      request.type_filling = ORDER_FILLING_FOK;
   else
      request.type_filling = ORDER_FILLING_RETURN;

   if(!OrderSend(request, result))
   {
      Print("SELL failed: ", result.retcode, " - ", result.comment);
      return 0.0;
   }

   if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
   {
      entryPrice = result.price;
      positionTicket = result.deal;
      Print("SELL executed @ ", result.price, " | Ticket: ", result.deal);
      return result.price;
   }

   Print("SELL retcode: ", result.retcode);
   return 0.0;
}

//+------------------------------------------------------------------+
//| Close current position                                            |
//+------------------------------------------------------------------+
double ClosePosition()
{
   // Find our position
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      double profit = PositionGetDouble(POSITION_PROFIT);
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
      request.comment   = "RLBridge CLOSE";

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
      if((fillType & SYMBOL_FILLING_IOC) != 0)
         request.type_filling = ORDER_FILLING_IOC;
      else if((fillType & SYMBOL_FILLING_FOK) != 0)
         request.type_filling = ORDER_FILLING_FOK;
      else
         request.type_filling = ORDER_FILLING_RETURN;

      if(OrderSend(request, result))
      {
         if(result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_PLACED)
         {
            Print("CLOSE executed @ ", result.price, " | P/L: ", profit);
            return profit;
         }
      }

      Print("CLOSE failed: ", result.retcode);
   }

   return 0.0;
}

//+------------------------------------------------------------------+
//| Update position tracking from MT5's position list                 |
//+------------------------------------------------------------------+
void UpdatePositionState()
{
   currentPosition = 0;
   entryPrice = 0.0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      currentPosition = (type == POSITION_TYPE_BUY) ? 1 : -1;
      entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
      positionTicket = ticket;
      return;
   }
}

//+------------------------------------------------------------------+
//| Get unrealized P/L of current position                            |
//+------------------------------------------------------------------+
double GetUnrealizedPnl()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;

      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != MagicNumber) continue;

      return PositionGetDouble(POSITION_PROFIT);
   }
   return 0.0;
}

//+------------------------------------------------------------------+
//| Handle reset signal from Python (new episode)                     |
//+------------------------------------------------------------------+
void HandleReset()
{
   Print("Reset signal received. Closing positions and restarting episode.");

   // Close any open position
   if(currentPosition != 0)
      ClosePosition();

   currentPosition = 0;
   entryPrice = 0.0;
   barCount = 0;
   episodeDone = false;

   // Delete reset file
   FileDelete(resetFile, FILE_COMMON);

   // Write acknowledgment
   WriteResult(0, 0.0, 0.0, AccountInfoDouble(ACCOUNT_BALANCE),
               AccountInfoDouble(ACCOUNT_EQUITY), 0, false);
}
//+------------------------------------------------------------------+
