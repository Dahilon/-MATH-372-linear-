const fs = require('fs');

function transpose(A){return A[0].map((_,j)=>A.map(r=>r[j]));}
function mm(A,B){const BT=transpose(B);return A.map(r=>BT.map(c=>r.reduce((s,x,i)=>s+x*c[i],0)));}
function mv(A,v){return A.map(r=>r.reduce((s,x,i)=>s+x*v[i],0));}
function solve(A,b){const n=A.length,M=A.map((r,i)=>r.slice().concat([b[i]]));for(let k=0;k<n;k++){let p=k;for(let i=k+1;i<n;i++)if(Math.abs(M[i][k])>Math.abs(M[p][k]))p=i;[M[k],M[p]]=[M[p],M[k]];const q=M[k][k];for(let j=k;j<=n;j++)M[k][j]/=q;for(let i=0;i<n;i++){if(i===k)continue;const f=M[i][k];for(let j=k;j<=n;j++)M[i][j]-=f*M[k][j];}}return M.map(r=>r[n]);}
function fit(X,y,lambda=0){const Xt=transpose(X),A=mm(Xt,X).map((r,i)=>r.map((v,j)=>v+(i===j&&i>0?lambda:0)));return solve(A,mv(Xt,y));}
function pred(X,b){return mv(X,b);}
function mse(a,b){return a.reduce((s,v,i)=>s+(v-b[i])**2,0)/a.length;}
function r2adj(y,yhat,p){const sst=y.reduce((s,v)=>s+(v-mean(y))**2,0),sse=mse(y,yhat)*y.length;return 1-(sse/sst)*(y.length-1)/(y.length-p-1);}
function combos(a,k,start=0,cur=[],out=[]){if(cur.length===k){out.push(cur.slice());return out;}for(let i=start;i<a.length;i++)combos(a,k,i+1,cur.concat(i),out);return out;}
function withIntercept(rows,cols){return rows.map(r=>[1,...cols.map(c=>r[c])]);}
function mean(a){return a.reduce((s,v)=>s+v,0)/a.length;}
function sd(a){const m=mean(a);return Math.sqrt(mean(a.map(v=>(v-m)**2)));}
function standardize(X){const p=X[0].length,mu=[],sc=[];for(let j=0;j<p;j++){const z=X.map(r=>r[j]);mu.push(mean(z));sc.push(sd(z)||1);}return {Z:X.map(r=>r.map((v,j)=>(v-mu[j])/sc[j])),mu,sc};}
function lasso(X,y,lambda){const n=X.length,p=X[0].length;const s=standardize(X),Z=s.Z;let b=Array(p).fill(0),a=mean(y);for(let it=0;it<10000;it++){let old=b.slice();let r=y.map((v,i)=>v-a-Z[i].reduce((q,x,j)=>q+x*b[j],0));a=mean(y)-mean(Z.map((r,i)=>r.reduce((q,x,j)=>q+x*b[j],0)));for(let j=0;j<p;j++){const zj=Z.map(r=>r[j]),rho=zj.reduce((q,x,i)=>q+x*(y[i]-a-Z[i].reduce((u,v,k)=>u+(k===j?0:v*b[k]),0)),0);const den=zj.reduce((q,x)=>q+x*x,0);b[j]=Math.sign(rho)*Math.max(Math.abs(rho)-lambda,0)/den;}if(Math.max(...b.map((v,j)=>Math.abs(v-old[j])))<1e-7)break;}return {intercept:a-coeffMeanCorrection(b,s),coef:b.map((v,j)=>v/s.sc[j])};}
function coeffMeanCorrection(b,s){return b.reduce((q,v,j)=>q+v*s.mu[j]/s.sc[j],0);}
function lpredict(X,m){return X.map(r=>m.intercept+r.reduce((s,v,j)=>s+v*m.coef[j],0));}
function parseCSV(path){const lines=fs.readFileSync(path,'utf8').trim().split(/\r?\n/);const head=lines[0].split(',');return {head,rows:lines.slice(1).map(x=>x.split(','))};}

const college=parseCSV('College.csv');
const h=college.head;const idx=Object.fromEntries(h.map((x,i)=>[x,i]));
const data=college.rows.map(r=>({name:r[0],private:r[idx.Private]==='Yes'?1:0, vals:h.slice(2).map(x=>+r[idx[x]])}));
const names=h.slice(2);const y=data.map(d=>d.vals[names.indexOf('Room.Board')]);
const xnames=['Accept','Enroll','Outstate','Books','Grad.Rate'];
const xidx=xnames.map(x=>names.indexOf(x));
const five=data.map(d=>xidx.map(j=>d.vals[j]));
console.log('Q4b best subset on five predictors (adjusted R2):');
for(let k=0;k<=5;k++){let best=null;for(const cs of combos([0,1,2,3,4],k)){const X=withIntercept(five,cs),b=fit(X,y),v=r2adj(y,pred(X,b),k);if(!best||v>best.adj)best={cs,b,adj:v};}console.log(k,best.cs.map(j=>xnames[j]).join('+')||'Intercept',best.adj.toFixed(5));}
const split=[...Array(data.length).keys()];let seed=42;function rnd(){seed=(1664525*seed+1013904223)%4294967296;return seed/4294967296;}split.sort(()=>rnd()-.5);const ntr=Math.floor(.75*data.length),tr=split.slice(0,ntr),te=split.slice(ntr);
// For parts (d)-(f), the 17 predictors are every College variable except Room.Board.
const remaining=names.filter(x=>x!=='Room.Board');
const rx=remaining.map(x=>names.indexOf(x));
const Xall=data.map(d=>[d.private,...rx.map(j=>d.vals[j])]);const Xtr=tr.map(i=>Xall[i]),Xte=te.map(i=>Xall[i]),ytr=tr.map(i=>y[i]),yte=te.map(i=>y[i]);
const bOLS=fit(Xtr.map(r=>[1,...r]),ytr);console.log('\nQ4d/e split:',ntr,'train /',te.length,'test');console.log('OLS test MSE',mse(yte,pred(Xte.map(r=>[1,...r]),bOLS)).toFixed(2));
let bestR={mse:Infinity},bestL={mse:Infinity};for(let t=-4;t<=6;t+=.1){const lam=Math.pow(10,t);const br=fit(Xtr.map(r=>[1,...r]),ytr,lam);const mr=mse(yte,pred(Xte.map(r=>[1,...r]),br));if(mr<bestR.mse)bestR={lam,mse:mr,b:br};const bl=lasso(Xtr,ytr,lam);const ml=mse(yte,lpredict(Xte,bl));if(ml<bestL.mse)bestL={lam,mse:ml,b:bl};}
console.log('Ridge best lambda',bestR.lam.toFixed(5),'test MSPE',bestR.mse.toFixed(2));console.log('Ridge coefficients',bestR.b.map(v=>v.toFixed(4)).join(', '));console.log('Lasso best lambda',bestL.lam.toFixed(5),'test MSPE',bestL.mse.toFixed(2));console.log('Lasso coefficients',bestL.b.coef.map(v=>v.toFixed(4)).join(', '));console.log('Lasso nonzero',bestL.b.coef.map((v,i)=>Math.abs(v)>1e-5?remaining[i]:'').filter(Boolean).join(', '));

// A compact pairwise scatterplot used in the written response.
function svgPair(){const vars=xnames.concat(['Room.Board']),W=900,C=150,pad=25;let s=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${W}" viewBox="0 0 ${W} ${W}"><rect width="100%" height="100%" fill="white"/>`;for(let a=0;a<vars.length;a++)for(let b=0;b<vars.length;b++){const va=data.map(d=>d.vals[names.indexOf(vars[a])]),vb=data.map(d=>d.vals[names.indexOf(vars[b])]);const x0=b*C+pad,y0=a*C+pad,xs=va.map(v=>x0+(v-Math.min(...va))/(Math.max(...va)-Math.min(...va))*100),ys=vb.map(v=>y0+100-(v-Math.min(...vb))/(Math.max(...vb)-Math.min(...vb))*100);s+=`<rect x="${x0}" y="${y0}" width="100" height="100" fill="#fafafa" stroke="#ddd"/>`+xs.map((x,i)=>`<circle cx="${x.toFixed(1)}" cy="${ys[i].toFixed(1)}" r="1.2" fill="#245" opacity=".32"/>`).join('');if(a===vars.length-1)s+=`<text x="${x0+50}" y="${y0+120}" text-anchor="middle" font-size="10">${vars[b]}</text>`;if(b===0)s+=`<text x="${x0-4}" y="${y0+55}" text-anchor="end" font-size="10">${vars[a]}</text>`;}return s+'</svg>';}
fs.writeFileSync('College_pairplot.svg',svgPair());
