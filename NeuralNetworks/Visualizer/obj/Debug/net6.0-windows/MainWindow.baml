   M S B A M L   `   `   `  ????[  WLiveChartsCore.SkiaSharpView.WPF, Version=2.0.0.0, Culture=neutral, PublicKeyToken=null}Xclr-namespace:LiveChartsCore.SkiaSharpView.WPF;assembly=LiveChartsCore.SkiaSharpView.WPF LiveChartsCore.SkiaSharpView.WPF  5   	   E AVisualizer, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null?$clr-namespace:Visualizer.NetworkInfoVisualizer.NetworkInfo 5   	   'clr-namespace:Visualizer
Visualizer 5   	   9!clr-namespace:Visualizer.TrainingVisualizer.Training 5	   	      Visualizer.MainWindow   5   	   W SPresentationCore, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35e aSystem.Windows.Controls.Ribbon, Version=6.0.2.0, Culture=neutral, PublicKeyToken=b77a5c561934e089R NWindowsBase, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35^ ZWindowsFormsIntegration, Version=6.0.2.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35H 9http://schemas.microsoft.com/winfx/2006/xaml/presentation      5   	   R NSystem.Xaml, Version=6.0.2.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   hlvcXclr-namespace:LiveChartsCore.SkiaSharpView.WPF;assembly=LiveChartsCore.SkiaSharpView.WPF     5   	   5info$clr-namespace:Visualizer.NetworkInfo     5   	   *localclr-namespace:Visualizer     5   	   6training!clr-namespace:Visualizer.Training     5	   	     ? Title$  
MainWindow??5   	   $	??900??6   $
??1800??6)    ? 	Resources 5         TrainingTemplate*  $Visualizer.Training.TrainingDataView  NetworkTemplate(  "Visualizer.NetworkInfo.NetworkInfo??%d   &	        &	 (     ?? 5   
   .R?5        5      ?? 5   
   .R?5        5      5      .??5      ?  ? RowDefinitions 5   
   ?? 5      $??2* ?6   ?? 5      $??5* ?6    ? ColumnDefinitions 5   
   ?? 5      $??2* ?6   ?? 5      $??3* ?6   ?? 5      $??3* ?6   ?? 5      $??3* ?6   .+?5   
   ?? $??1q?6    ?? CornerRadius$ 5??6%   
???   ?$??10q?6J    ?? Padding$ 10q?6V   ._?5       ?? .+?5!      ?? $
??Data??6   $??24(?6(   $
??Bold"?66   ?? 5"      #??[  6!   $??Center=?6r   ??6U   ?? *
Training+6r   ?? 5#       ?? Orientation$ 
Horizontal=?6   .+?5$      ?? $??10q?6@   $??5q?6L   $
??Left=?6X   ??6   ?? *GetDataCommand+.??6s   
Get data6}   ?? 5%      $??10q?6     ?? Padding$ 5q?6,   ??68   ?? *
DataInfo+5&      5'      5(      ? 5)   
   $??1??6    5*      ?? 5+      $??2* ?6#   ?? 5,      $??* ?6#   .+?5.      ?? $??1q?6   $ 5??6)   
???   ?$??10q?6N   $ 10q?6Z   $??0??6g   ._?5/      ?? .+?50      ?? $??Train??6    $??24(?6-   $
??Bold"?6;   ?? 51      $ 
Horizontal=?6!   $
??0,10q?6:   .+?52      ?? $??
Train for ??6$   ?? 53      $??50??6;   ??6"   ?? *	EpochNb+6;   ?? 54      $?? epochs??6$   55      ?? 56      $ 
Horizontal=?6!   $
??0,10q?6:   .+?57      ?? $??Validate every ??6$   ?? 58      $??50??6L   ??6"   ?? *EpochsBetweenValidations+6L   ?? 59      $??	 epoch(s)??6$   5:      ?? 5;      $ 
Horizontal=?6!   $
??0,10q?6:   .+?5<      ?? $
??0,10q?6?   ??6$   ?? *	TrainInfo+6?   ?? 5=      $??Center=?6!   $??Center=?6<   $
??10,0q?6w   $??5q?6?   ??6Y   ?? *	GoCommand+.??6?   Train!6?   ?? 5>      $
??0,10q?6B   ??6$   ?? *CurrentEpoch+6B   5?      ?? 5@      $ 
Horizontal=?6!   $
??0,10q?6:   .+?5A      ?? $??Center=?6!   $??Center=?6<   $
??10,0q?6z   $??5q?6?   ??6Y   ?? *ClearCommand+.??6?   Clear graph6?   ?? 5B      $??Center=?6!   $??Center=?6<   $
??10,0q?6y   $??5q?6?   ??6Y   ?? *SaveCommand+.??6?   Save trained network6?   5C      5D      5E      5F      ?? 5G   
   $??1??6   $??1q?6   $ 5??62   
???   ?$??10q?6W   $ 10q?6c   ._?5H      ?  5I      ?? 5J      $	??90* ?6'   ?? 5K      $	??73* ?6'    5M      ?? 5N      $??* ?6$   ?? 5O      $??7* ?6$   .+?5Q      ?? $??2??6   .+?5R      ?? $??Network??6    $??24(?6/   $
??Bold"?6=   ?? 5S      $ 
Horizontal=?6!   .+?5T      ?? $??Center=?6!   $??10q?6\   $??5q?6h   $
??Left=?6t   ??6<   ?? *LoadCommand+.??6?   Load network6?   ?? 5U      $??Center=?6!   $??10q?6]   $??5q?6i   ??6<   ?? *ResetCommand+.??6u   Reset network6?   5V      5W      ?? 5X      #??[ 6!   $??1??6w   $??2??6?   ??6T   ?? *NetworkSummary+6?   5Y      5Z      5   /LiveChartsCore.SkiaSharpView.WPF.CartesianChart  5[   
   $??1??6<   $??3??6I   $??1??6]      Series 6   ?? *
Collection+6]   5\      5]      