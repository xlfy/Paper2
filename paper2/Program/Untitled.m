%%极限学习机%%

%Sample=xlsread('Iris.xlsx');
Sample=textread('housing.data.txt');

[n,m]=size(Sample);
X=Sample(:,1:size(Sample,2)-1)';
Y=Sample(:,size(Sample,2))';

Hidden_Num=1500;

%开始时间
start_time=cputime;
%初始化
W=rand(Hidden_Num,m-1)*2-1;
B=rand(Hidden_Num);
Node_output=W*X;
ind=ones(1,n);
Bias=B(:,ind);
Node_output=Node_output+Bias;
O=1 ./ (1 + exp(-Node_output));
OutputWeight=pinv(O') * Y';
end_time=cputime;
run_time=end_time-start_time;
Ye=(O' * OutputWeight)';  
error=sqrt(mse(Ye - Y));
x=(1:n);
plot(x,Y','-ob');
hold on;
plot(x,Ye','-r');