const fs = require('fs');

const answer = fs.readFileSync('Homework4_student_response.md', 'utf8');
const questions = `
<h1>Homework 4</h1>
<p><b>MATH 372 Linear Regression Analysis</b></p>
<p>This document contains the assignment questions followed by my student-style responses.</p>

<h2>Question 1 Ridge Regression</h2>
<p>Consider the regression model y = Xβ + ε, where the errors are independent and identically distributed as N(0, σ²). Let Z = XᵀX. Using the regularization definition of ridge regression with tuning parameter λ, show that E(β̂ Ridge) = (I + λZ⁻¹)⁻¹β. Based on this result, calculate the bias of β̂ Ridge. Finally, show that Var(β̂ Ridge) = σ²(I + λZ⁻¹)⁻¹Z⁻¹(I + λZ⁻¹)⁻¹.</p>

<h2>Question 2 Subset Selection</h2>
<p>Go through and code the lab in the Subset Selection Methods sections 6.5.1 and 6.5.2 of An Introduction to Statistical Learning. Type the code to make sure it works and turn in the code and output as part of the file.</p>

<h2>Question 3 Comparing Selection Methods</h2>
<p>Suppose that best subset, forward stepwise, and backward stepwise selection are performed for linear regression on one data set. The same criterion is used to select the best model, and each approach produces p + 1 models containing 0, 1, ..., p predictors. Answer the following:</p>
<ol><li>Which method with exactly k predictors has the smallest training MSE?</li><li>Which method with exactly k predictors has the smallest test MSPE?</li><li>True or false: the predictors in the k-variable model identified by forward stepwise selection are a subset of the predictors in the (k + 1)-variable forward-stepwise model.</li><li>True or false: the predictors in the k-variable model identified by best subset selection are a subset of the predictors in the (k + 1)-variable best-subset model.</li></ol>

<h2>Question 4 College Data</h2>
<p>Download the College data set from https://trevorhastie.github.io/ISLR/College.csv. The data describe summary characteristics of American colleges and universities, including the University of San Francisco. Study which factors are important in determining Room.Board.</p>
<ol type="a"><li>Using Accept, Enroll, Outstate, Books, and Grad.Rate, make a pairwise scatterplot with Room.Board and comment on the trends.</li><li>Run best subset selection for Room.Board on those five predictors using adjusted R². State the selected model and comment on differences between the chosen models.</li><li>Perform ridge regression across a grid of λ values for Room.Board using the remaining 17 predictors, treating Private as a factor. Plot MSPE, refine the grid to find the minimum, state the final model and average MSPE, and compare ridge estimates with standard linear regression.</li><li>Repeat the ridge analysis using the Lasso.</li><li>Choose between the ridge and Lasso models and discuss variance, interpretability, inference, and prediction.</li></ol>
<hr>
<h1>Student Response</h1>
`;

function inline(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.*?)\*\*/g,'<b>$1</b>').replace(/`([^`]+)`/g,'<code>$1</code>');
}
function mdToHtml(md){
  const lines=md.split(/\r?\n/),out=[];let inCode=false,buf=[];
  for(const line of lines){
    if(line.startsWith('```')){if(inCode){out.push('<pre>'+buf.join('\n').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</pre>');buf=[];inCode=false;}else inCode=true;continue;}
    if(inCode){buf.push(line);continue;}
    if(!line.trim()){continue;}
    if(line.startsWith('# '))out.push('<h1>'+inline(line.slice(2))+'</h1>');
    else if(line.startsWith('## '))out.push('<h2>'+inline(line.slice(3))+'</h2>');
    else if(line.startsWith('### '))out.push('<h3>'+inline(line.slice(4))+'</h3>');
    else if(line.startsWith('|')){out.push('<p>'+inline(line)+'</p>');}
    else if(line.startsWith('- '))out.push('<p>• '+inline(line.slice(2))+'</p>');
    else out.push('<p>'+inline(line)+'</p>');
  } return out.join('\n');
}
const html=`<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:Arial,sans-serif;margin:0.8in;color:#111;font-size:11pt;line-height:1.35}h1{font-size:22pt;margin-top:0.25in;border-bottom:1px solid #777;padding-bottom:6px}h2{font-size:16pt;margin-top:0.3in}h3{font-size:13pt;margin-top:0.22in}pre{font-family:Menlo,monospace;font-size:8.5pt;background:#f4f4f4;padding:10px;white-space:pre-wrap;border:1px solid #ddd}code{font-family:Menlo,monospace;font-size:9pt}hr{border:0;border-top:1px solid #aaa;margin:0.35in 0}li{margin:5px 0}</style></head><body>${questions}${mdToHtml(answer)}</body></html>`;
fs.writeFileSync('Homework 4.html',html);
