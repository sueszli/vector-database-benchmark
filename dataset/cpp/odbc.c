//2023.06.07 suppress debug for return other than SQL_NO_DATA for unbound data
//2023.01.27 report errors from SQLFetch, set locale from environment defaults for drivers which do not call setlocale by default (e.g. exasol odbc)
//2022.06.30 fix variable width column data before fixed width could be truncated, e.g. select cast('important string' AS varchar(max)) col1, 1 col2
//2017.12.11 added support for SQLGUID
//2017.07.11 added support for Time(type#-154) and DatetimeOFFSET(type#-155), the latter with timezone info ignored
//2015.06.19 fix for unixodbc 2.3, and indeterminate column size
//2013.05.07 added timeout(in Secs). e.g. h:.odbc.open(`northwind;timeout);.odbc.eval[(h;timeout);"select * from Orders"]
//2011.03.11 64bit was not detecting SQL_NULL_DATA correctly
//2010.07.06 support mssql datetime (sql_type_date.. are 91 92 93 so map to 9 10 11
//2010.01.18 SQLInteger->SQLULEN for v64 build
//2008.03.24 load "" instead of () for null text
//2007.05.04 fix null charstrings;read bool and byte as smallint
//2007.03.28  KB null->0  KS(>8)->0
//cl -DWIN32  /LD /Oy odbc.c odbc.def q.lib odbc32.lib user32.lib
//cl -D_WIN64 /LD /Oy odbc.c odbc.def q.lib odbc32.lib user32.lib bufferoverflowU.lib
//gcc -shared ../c/odbc.c -o odbc.so -lodbc -fPIC
//usr/local/gcc-3.3.2/bin/gcc -G ../c/odbc.c -o odbc.so -lodbc   /-lodbcinst  [-m64 -fPIC]
#include<string.h>
#include<locale.h>
#include"d.h" // http://www.unixodbc.org
V free(V*);
V sl(){ZI i;if(!i)setlocale(LC_ALL,""),i=1;}
ZS err(I f,D d){ZC e[1006];I i;H j;SQLError(0,f?0:d,f?d:0,e,(SQLINTEGER*)&i,e+6,1000,&j);R e[5]=' ',e;}
#define Q(x,s) P(x,krr(s))
#define Q0(x) {I r=(x);if(r){S s=err(0,d);if(r!=1)R       krr(s);if(*s)O("%s\n",s);}}
#define QC(x,_c) {I r=(x);if(r){S s=err(1,d);if(r!=1)R _c,d0(d),krr(s);if(*s)O("%s\n",s);}}
#define CLEAN(n) clean(x,y,b,tb,wb,nb,n)
ZV clean(K x,K y,S*b,H*tb,SQLINTEGER*wb,SQLLEN*nb,SQLINTEGER j){if(x)r0(x);if(y)r0(y);if(j!=-1){DO(j,if(wb[i])free(b[i]));free(b),free(tb),free(wb),free(nb);}}
K2(open){Z D d9;H j=xt==KS;D d,v=GetForegroundWindow();ZC b[1024];sl();
 if(!d9)SQLAllocEnv(&d9);Q0(SQLAllocConnect(d9,&d))Q(y->t!=-KJ||!j&&xt!=-KS&&xt!=KC,(S)"type")Q(j&&xn!=3,(S)"length")
 if(y->j)SQLSetConnectAttr(d,SQL_LOGIN_TIMEOUT,(SQLPOINTER)(SQLULEN)y->j,0);
 Q0(xt==KC?SQLDriverConnect(d,v,xG,(H)xn,b,(H)1024,&j,v?SQL_DRIVER_COMPLETE:SQL_DRIVER_NOPROMPT):
  SQLConnect(d,j?*xS:xs,S0,j?xS[1]:0,S0,j?xS[2]:0,S0))R kj((J)(L)d);}
K1(close){D d=(D)(L)xj;R SQLDisconnect(d),SQLFreeConnect(d),knk(0);}ZV d0(D d){SQLFreeStmt(d,SQL_DROP);}
Z D d1(J j){D d=(D)(L)j;Q0(SQLAllocStmt(d,&d))R d;}
ZK rs(I a,D d,H j){C b[128];SQLLEN n=128;K x=ktn(a?0:KS,0);if(!a)for(SQLBindCol(d,j,SQL_C_CHAR,b,n,&n);!SQLFetch(d);)js(&x,ss(b));R d0(d),x;}
ZK fk(K x,S s,H j){D d=d1(xj);U(d)R rs(SQLForeignKeys(d,(S)0,0,(S)0,0,(S)0,0,(S)0,0,(S)0,0,s,S0),d,j);}
ZK tv(K x,S s){D d=d1(xj);U(d)R rs(SQLTables(d,(S)0,0,(S)0,0,(S)0,0,s,S0),d,3);}
K2(keys){D d=d1(xj);U(d)R rs(SQLPrimaryKeys(d,(S)0,0,(S)0,0,y->s,S0),d,4);}	K1(tables){R tv(x,(S)"TABLE,SYNONYMS");}
K2(fkeys){K r=fk(x,y->s,8);R r&&r->t?knk(2,r,fk(x,y->s,3)):r;}			    K1(views){R tv(x,(S)"VIEW");}
ZI ds(S s){DATE_STRUCT*d=(DATE_STRUCT*)s;R ymd(d->year,d->month,d->day);} ZS dtb(S s,I n){for(;n--&&s[n]==' ';);R sn(s,n+1);}
ZI vs(S s){TIME_STRUCT*t=(TIME_STRUCT*)s;R 3600*t->hour+60*t->minute+t->second;} ZJ const j0=86400000000000LL,j1=1000000000LL;
ZJ ns(S s){SS_TIME2*t=(SS_TIME2*)s;R j1*(3600*t->hour+60*t->minute+t->second)+t->fraction;}
ZJ ps(S s){SS_TIMESTAMPOFFSET*t=(SS_TIMESTAMPOFFSET*)s;R j0*ymd(t->year,t->month,t->day)+j1*(3600*t->hour+60*t->minute+t->second)+t->fraction;}
V na(J n,G*a){G j;DO(n--/2,j=a[n-i];a[n-i]=a[i];a[i]=j)}
ZI ut[]={0,KS,KF,KF,KI,KH,KF,KE,KF,KD,KV,KZ,KS,0,0,0,0, KJ,KH,KH,KS,KS,0,2,KN,KP};//KG,KB  use KH for the nulls
ZI wt[]={0, 0, 8, 8, 4, 2, 8, 4, 8, 6, 6,16, 0,0,0,0,0,  8, 1, 1, 0, 0,0,16,12,20};
ZH ct[]={0, 1, 8, 8, 4, 5, 8, 7, 8, 9,10,11, 1,1,0,0,0,-25,-6,-7, 1, 1,1,-11,-2,-2};// -5/-25(odbc 2/3)
ZS nu(I t){ZF f;ZE e;ZJ j=nj;ZI i=ni;ZH h=nh;ZC g;ZS ns;Z U u;if(!ns)f/=f,e=f,ns=ss((S)"");R t==KS?(S)&ns:t==KF||t==KZ?(S)&f:t==KE?(S)&e:t==KJ||t==KN||t==KP?(S)&j:t==KH?(S)&h:t==KG||t==KB?(S)&g:t==2?(S)&u:(S)&i;}
ZK gb(D d,H j,I t){H c=ct[t],g=c?c:-2,m=512;K x=ktn(c?KC:KG,m),y=ktn(xt,0);SQLLEN n=0;SQLRETURN r;while(1){r=SQLGetData(d,j,g,kG(x),xn=m,&n);if(SQL_SUCCEEDED(r)&&n!=SQL_NULL_DATA)xn=n==SQL_NO_TOTAL||n>xn?xn:n,xn-=xn&&c&&!kG(x)[xn-1],jv(&y,x);else /*if(r==SQL_NO_DATA)*/break;}r0(x);R y;}
K eval(K x,K y,K z){K*k;S*b,s;SQLULEN w;SQLLEN*nb;SQLINTEGER*wb;SQLRETURN sr;H*tb,u,t,j=0,p,m,ubi=0,bi=0;F f;J v;C c[128];I n=xj<0;D d=d1(n?-xj:xj);U(d)x=y;Q(z->t!=-KJ||xt!=-KS&&xt!=KC,(S)"type")
 if(z->j)SQLSetStmtAttr(d,SQL_ATTR_QUERY_TIMEOUT,(SQLPOINTER)(SQLULEN)z->j,0);
 if(xt==-KS)QC(SQLColumns(d,(S)0,0,(S)0,0,xs,S0,(S)0,0),-1)else{I e;K q=kpn(xG,xn);ja(&q,(S)"\0");QC(SQLExecDirect(d,q->G0,xn),r0(q));}
 SQLNumResultCols(d,&j);P(!j,(d0(d),knk(0)))
 b=malloc(j*SZ),tb=malloc(j*2),wb=malloc(j*SZ),nb=memset(malloc(j*SZ),0,j*SZ),x=ktn(KS,j),y=ktn(0,j);
 DO(j,QC(SQLDescribeCol(d,(H)(i+1),c,128,&u,&t,&w,&p,&m),CLEAN(0))xS[i]=sn(c,u);
 if(t>90)t-=82;if(t<-153)t+=142;Q(t<-13||t>12,xS[i])wb[i]=ut[tb[i]=t=t>0?t:12-t]==KS&&w?w+1:wt[t];if(ut[t]==KS&&(n||!wb[i]||wb[i]>9))tb[i]=13;ubi=wb[i]||ubi?ubi:i+1;bi=wb[i]?i+1:bi)
 bi=!ubi||!bi||ubi>bi; // use binding if no bound past unbound (driver suppported SQL_GETDATA_EXTENSIONS:GD_ANY_COLUMN is nyi)
 DO(j,kK(y)[i]=ktn(ut[t=tb[i]],0);if((w=wb[i])){b[i]=malloc(w);if(bi){QC(SQLBindCol(d,(H)(i+1),ct[t],b[i],w,nb+i),CLEAN(i+1));}})
 for(;SQL_SUCCEEDED(sr=SQLFetch(d));)DO(j,k=kK(y)+i;u=ut[t=tb[i]];if(!bi&&wb[i])QC(SQLGetData(d,(H)i+1,ct[t],b[i],wb[i],nb+i),CLEAN(j));s=b[i];n=SQL_NULL_DATA==(int)nb[i];if(!u)jk(k,n?ktn(ct[t]?KC:KG,0):wb[i]?kp(s):gb(d,(H)(i+1),t));else ja(k,n?nu(u):u==KH&&wb[i]==1?(t=(H)*s,(S)&t):u==KS?(s=dtb(s,nb[i]),(S)&s):u==2?na(4,s),na(2,s+4),na(2,s+6),s:u==KN?(v=ns(s),(S)&v):u==KP?(v=ps(s),(S)&v):u<KD?s:u==KZ?(f=ds(s)+(vs(s+6)+*(I*)(s+12)/1e9)/8.64e4,(S)&f):(w=u==KD?ds(s):vs(s),(S)&w)))
 QC(sr==SQL_NO_DATA?0:sr,CLEAN(j));
 if(!SQLMoreResults(d))O("more\n");R clean(0,0,b,tb,wb,nb,j),d0(d),xT(xD(x,y));}
/*
#define SQL_CHAR            1
#define SQL_NUMERIC         2
#define SQL_DECIMAL         3
#define SQL_INTEGER         4
#define SQL_SMALLINT        5
#define SQL_FLOAT           6
#define SQL_REAL            7
#define SQL_DOUBLE          8
#define SQL_DATE            9
#define SQL_TIME           10
#define SQL_TIMESTAMP      11
#define SQL_VARCHAR        12
#define SQL_LONGVARCHAR     (-1)
#define SQL_BINARY          (-2)
#define SQL_VARBINARY       (-3)
#define SQL_LONGVARBINARY   (-4)
#define SQL_BIGINT          (-5) -25
#define SQL_TINYINT         (-6)
#define SQL_BIT             (-7)
#define SQL_WCHAR		 	(-8)
#define SQL_WVARCHAR	 	(-9)
#define SQL_WLONGVARCHAR 	(-10)
guid                         -11
ZI un(D d,H j){I u;SQLColAttribute(d,j,SQL_DESC_UNSIGNED,0,0,0,&u);R u;}
*/

