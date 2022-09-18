
%% Main Nonlinear fiting

close all;
clear;
clc;


%% Data

Filter={'*.csv;*.xlsx;*.xls'};
[FileName, FilePath]=uigetfile(Filter);
if FileName==0
    return;
end
FullFileName=[FilePath FileName];
X = xlsread(FullFileName);

%% Plot

boxplot(X(2:end), 'PlotStyle', 'traditional', 'Widths',0.3)
