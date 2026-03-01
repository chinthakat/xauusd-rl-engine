//+------------------------------------------------------------------+
//| TickExporter.mq5 — Export M1 Bar Data from Strategy Tester      |
//|                                                                  |
//| Run this EA in MT5 Strategy Tester to capture completed M1 bars |
//| with OHLCV, spread, and MA indicators.                          |
//| Output goes to Common\Files\tick_export\.                        |
//|                                                                  |
//| Strategy Tester Settings:                                        |
//|   - Mode: "Every tick" or "1 Minute OHLC"                       |
//|   - Symbol: XAUUSD                                               |
//|   - Period: M1                                                    |
//|   - Date range: 1 month                                          |
//+------------------------------------------------------------------+
#property copyright "XAU Learning Model"
#property version   "2.00"
#property strict

//--- Input parameters
input string   ExportFolder   = "tick_export";   // Output subfolder in Common\Files
input string   ExportFilename = "";              // Auto-generated if empty
input bool     IncludeIndicators = true;         // Include MA indicators
input int      MA_Fast        = 5;               // Fast MA period
input int      MA_Medium      = 10;              // Medium MA period (MA10)
input int      MA_Cross       = 20;              // Cross MA period (MA20)
input int      MA_Slow        = 100;             // Slow MA period

//--- Global variables
int    fileHandle = INVALID_HANDLE;
string filePath;
int    barCount = 0;
datetime lastBarTime = 0;

//--- Indicator handles
int    handleMA_Fast;
int    handleMA_Medium;
int    handleMA_Cross;
int    handleMA_Slow;

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // Build filename
   string fname = ExportFilename;
   if(fname == "")
   {
      fname = _Symbol + "_M1_export.csv";
   }

   filePath = ExportFolder + "\\" + fname;

   // Open file for writing (ANSI mode for Python compatibility)
   fileHandle = FileOpen(filePath, FILE_WRITE | FILE_ANSI | FILE_COMMON);
   if(fileHandle == INVALID_HANDLE)
   {
      Print("ERROR: Cannot create export file: ", filePath, " Error: ", GetLastError());
      return(INIT_FAILED);
   }

   // Write CSV header
   string header = "timestamp,Open,High,Low,Close,Volume,Spread";

   if(IncludeIndicators)
      header += ",MA5,MA10,MA20,MA100";

   FileWriteString(fileHandle, header + "\n");

   // Initialize MAs
   if(IncludeIndicators)
   {
      handleMA_Fast   = iMA(_Symbol, PERIOD_M1, MA_Fast, 0, MODE_SMA, PRICE_CLOSE);
      handleMA_Medium = iMA(_Symbol, PERIOD_M1, MA_Medium, 0, MODE_SMA, PRICE_CLOSE);
      handleMA_Cross  = iMA(_Symbol, PERIOD_M1, MA_Cross, 0, MODE_SMA, PRICE_CLOSE);
      handleMA_Slow   = iMA(_Symbol, PERIOD_M1, MA_Slow, 0, MODE_SMA, PRICE_CLOSE);

      if(handleMA_Fast == INVALID_HANDLE || handleMA_Medium == INVALID_HANDLE ||
         handleMA_Cross == INVALID_HANDLE || handleMA_Slow == INVALID_HANDLE)
      {
         Print("ERROR: Failed to create MA indicators");
         return(INIT_FAILED);
      }
   }

   Print("TickExporter v2 (M1 bars). Output: Common\\Files\\", filePath);
   Print("Symbol: ", _Symbol, " | Point: ", _Point, " | Digits: ", _Digits);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(fileHandle != INVALID_HANDLE)
   {
      FileClose(fileHandle);
      Print("TickExporter finished. Total bars exported: ", barCount,
            " | File: ", filePath);
   }

   if(IncludeIndicators)
   {
      if(handleMA_Fast != INVALID_HANDLE)   IndicatorRelease(handleMA_Fast);
      if(handleMA_Medium != INVALID_HANDLE) IndicatorRelease(handleMA_Medium);
      if(handleMA_Cross != INVALID_HANDLE)  IndicatorRelease(handleMA_Cross);
      if(handleMA_Slow != INVALID_HANDLE)   IndicatorRelease(handleMA_Slow);
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   if(fileHandle == INVALID_HANDLE) return;

   // Only process on new bar (write the COMPLETED previous bar)
   datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
   if(currentBarTime == lastBarTime) return;

   // Skip very first tick (no completed bar yet)
   if(lastBarTime == 0)
   {
      lastBarTime = currentBarTime;
      return;
   }

   lastBarTime = currentBarTime;
   barCount++;

   // Get completed bar data (bar index 1 = most recently closed bar)
   datetime barTime = iTime(_Symbol, PERIOD_M1, 1);
   double   barOpen  = iOpen(_Symbol, PERIOD_M1, 1);
   double   barHigh  = iHigh(_Symbol, PERIOD_M1, 1);
   double   barLow   = iLow(_Symbol, PERIOD_M1, 1);
   double   barClose = iClose(_Symbol, PERIOD_M1, 1);
   long     barVol   = iVolume(_Symbol, PERIOD_M1, 1);

   // Get current spread in points
   double spread_pts = (SymbolInfoDouble(_Symbol, SYMBOL_ASK) -
                        SymbolInfoDouble(_Symbol, SYMBOL_BID)) / _Point;

   // Build CSV line
   string line = TimeToString(barTime, TIME_DATE|TIME_MINUTES|TIME_SECONDS);
   line += "," + DoubleToString(barOpen, _Digits);
   line += "," + DoubleToString(barHigh, _Digits);
   line += "," + DoubleToString(barLow, _Digits);
   line += "," + DoubleToString(barClose, _Digits);
   line += "," + IntegerToString(barVol);
   line += "," + DoubleToString(spread_pts, 1);

   // MA indicators (on completed bar)
   if(IncludeIndicators)
   {
      double maF[1], maM[1], maC[1], maS[1];

      if(CopyBuffer(handleMA_Fast, 0, 1, 1, maF) == 1)
         line += "," + DoubleToString(maF[0], _Digits);
      else
         line += ",";

      if(CopyBuffer(handleMA_Medium, 0, 1, 1, maM) == 1)
         line += "," + DoubleToString(maM[0], _Digits);
      else
         line += ",";

      if(CopyBuffer(handleMA_Cross, 0, 1, 1, maC) == 1)
         line += "," + DoubleToString(maC[0], _Digits);
      else
         line += ",";

      if(CopyBuffer(handleMA_Slow, 0, 1, 1, maS) == 1)
         line += "," + DoubleToString(maS[0], _Digits);
      else
         line += ",";
   }

   FileWriteString(fileHandle, line + "\n");

   // Flush periodically
   if(barCount % 1000 == 0)
   {
      FileFlush(fileHandle);
      Print("Exported ", barCount, " M1 bars...");
   }
}
//+------------------------------------------------------------------+
