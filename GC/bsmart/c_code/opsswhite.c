#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "EEGdef.h"
#include "EEGmat.h"

/*****
   Write residuals probabilities to RESID.OUT(ascii file) 

   with the same dat format as 
   input datfile which can be read using readdat.m, but the length 
   of data in each channel is WIN - ORDER  

Change model order defined in EEGdef.h, 02/17/99

********/

void MAR_residual(double **xin, double **rout, double **A,
        int nchns, int order, int xinlength);
int main(int argc, char**argv)
{ 
  int NPTS =0;
  int NCHN =0;
  int NTRLS =0;
  int WIN=0;
  int MODORDER=0;
  if (argc==9)
   {
     NCHN=atoi(argv[4]);
     NTRLS=atoi(argv[5]);
     NPTS=atoi(argv[6]);
     WIN=atoi(argv[7]);
     MODORDER=atoi(argv[8]);
   }
  else if ( argc!=4 && argc!=9)
	{
	  fprintf(stderr,"opsswhite datfile A Ve Nchannels Ntrails Npoints WIN MODORDER\n");
	  exit(1);
	}
  else { 
  double chan[1],trai[1],poin[1],wind[1],order[1];
  FILE *chanfp,*traifp,*poinfp,*windfp,*orderfp;
  if((chanfp=fopen("channel","r"))==NULL)   
              printf("The file'channel' was not opened\n");   
  else   
    fscanf(chanfp,"%lf",&chan[0]);  
    NCHN=(int)chan[0];
    fclose(chanfp);
  if((traifp=fopen("trail","r"))==NULL)   
              printf("The file'trail' was not opened\n");   
  else   
    fscanf(traifp,"%lf",&trai[0]);  
    NTRLS=(int)trai[0];
    fclose(traifp);
  if((poinfp=fopen("points","r"))==NULL)   
              printf("The file'points' was not opened\n");   
  else   
    fscanf(poinfp,"%lf",&poin[0]);  
    NPTS=(int)poin[0];
    fclose(poinfp);
  if((windfp=fopen("window","r"))==NULL)   
              printf("The file'window' was not opened\n");   
  else   
    fscanf(windfp,"%lf",&wind[0]);  
    WIN=(int)wind[0];
    fclose(windfp);
  if((orderfp=fopen("order","r"))==NULL)   
              printf("The file'order' was not opened\n");   
  else   
    fscanf(orderfp,"%lf",&order[0]);  
    MODORDER=(int)order[0];
    fclose(orderfp);}

  FILE *inpt, *shfp, *stimp, *fp;
  FILE *fr;

  int *shift, *stim;
  int shfmin=0, shfmax=0, stimmin, stimmax;

  double *A[MAXORDER],*Ve,*tildA;
  float **dat; 
  double **x;   /*  rout is residual output */
  int *n;  /* n[j] is the number of data in j-th segment, 120 */  
  int i, j, k, rec, idx, t;

  double **rout;   /*  rout is residual output */
  double *xtemp[NCHN];
  double *rtemp[NCHN];

  double mean[NCHN], ss[NCHN], corfun[50], sigthresh, prob;
  int nlags = 5;
  int nresids = WIN-MODORDER;
  int l, lag, nsig=0, ntot=0;

  sigthresh = 2.000 / sqrt((double)nresids);
  
  if( argc<4)
	{
	  fprintf(stderr,"Usage: opss datfile A Ve\n");
	  exit(1);
	}


  for(i=0;i<MAXORDER;i++){
    if((A[i]=malloc(NCHN*NCHN*sizeof(double)))==NULL)
      EEGerror("main---memory allocation error\n");
  }
  if((tildA=malloc(MAXORDER*(MAXORDER+1)*NCHN*NCHN*sizeof(double)/2))==NULL)
    EEGerror("main---memory allocation error\n");
  if((Ve=malloc(NCHN*NCHN*sizeof(double)))==NULL)
    EEGerror("main---memory allocation error\n");


  /* allocation of memory for dat */
  dat = malloc(NCHN*sizeof(float*));
  for( i = 0; i < NCHN; i++)
    dat[i] = malloc(WIN*NTRLS*sizeof(float));

  x = malloc(NCHN*sizeof(double*));
  for( i = 0; i < NCHN; i++)
    x[i] = malloc(WIN*NTRLS*sizeof(double));

  /***  For residual calculation   ****/
  rout = malloc(NCHN*sizeof(double*));
  for( i = 0; i < NCHN; i++)
    rout[i] = malloc((WIN-MODORDER)*NTRLS*sizeof(double));
  /*
  rtemp = malloc(NCHN*sizeof(double*));
  for( i = 0; i < NCHN; i++)
    rtemp[i] = malloc((WIN-MODORDER)*sizeof(double));
	*/

  n=malloc(NTRLS*sizeof(int));
  shift = malloc(NTRLS*sizeof(int));
  stim = malloc(NTRLS*sizeof(int));


  if((inpt = fopen(argv[1],"rb")) == NULL) {
	printf("\007Error opening input file!\n"); return -1;
  } 
/*  if((shfp = fopen(argv[2],"rt")) == NULL) {
	printf("\007Error opening shift file!\n"); return;
  } 
  if((stimp = fopen(argv[3],"rt")) == NULL) {
	printf("\007Error opening stim file!\n"); return;
  } 
*/

  /* Initialization, required by MARfit */ 	
  for( i = 0; i < NTRLS; i++) n[i] = WIN;

  /*  for( i=0; i<NTRLS; i++ ) fscanf(shfp,"%d",&shift[i]); */
 /* for( i=0; i<NTRLS; i++ ) fscanf(shfp,"%d", &shift[i]); 
  fclose(shfp);
  minmax(NTRLS, shift, &shfmin, &shfmax);*/
  /*  printf("min, max = %d  %d\n", shfmin, shfmax); */

 /* for( i=0; i<NTRLS; i++ ) fscanf(stimp,"%d", &stim[i]); 
  fclose(stimp);
  minmax(NTRLS, stim, &stimmin, &stimmax);*/
  stimmin=0;  /* convert to points  */
  /*  printf("min, max = %d  %d\n", stimmin, stimmax); */


  /* printf("Trl, CHN, T, WIN = %d %d %d %d\n",NTRLS,NCHN, NPTS, WIN); */

  /*********  read monkey dat *********/

  t = -(stimmin - WIN/2)*5 - 5;   /* - (20-8)*5 +5 = -60 - 5 msec */    
  /*  for (rec=3; rec < 104; rec++) {  */
   for ( rec=abs(shfmin); rec < (NPTS-WIN-shfmax+1); rec++) { 
	t+=5;
	/*printf("index, t = %d  %d msec\n", (rec-abs(shfmin)+1), t);*/
    for( j = 0; j < NCHN; j++){
	  idx=0;
	  for( i = 0; i < NTRLS; i++){
		if(fseek(inpt, sizeof(float)*(shift[i] + rec + i*NCHN*NPTS + j*NPTS), 0) != 0){
		  printf("Error in fseek\n"); exit(-1);
		} 
		
		if( fread(&dat[j][idx],sizeof(float),WIN,inpt) !=WIN) {
		  if(feof(inpt)) 
			{printf("premature end of file in read data");exit(-1);}
		  else {printf("file read error");exit(-1);}
		}
		
		idx+=WIN;
	  }
	}

	/*
	  for( i = 0; i < 3; i++){
	  for( j = 0; j < WIN*NTRLS; j++) { 
	  printf("%f\n", dat[i][j]);
	  }
	  printf("OOOOOOOO\n");
	  }
	  */



/* convert data format from float to double  */
  for (i=0; i < NCHN; i++)
	for(j=0; j < WIN*NTRLS; j++)
	  x[i][j] = dat[i][j];

  MARfit(x,NCHN,n,NTRLS,MODORDER,tildA);
  EEGrealA(tildA,A,Ve,NCHN,MODORDER);
  /*
  MARfit(x,NCHN,n,NTRLS,6,tildA);
  EEGrealA(tildA,A,Ve,NCHN,5);
  */

  /******  calculate residual of MVAR for whiteness test  *******/
  /* First loop over trials, and then loop over channels & allocate memory for 
	 xtemp & rtemp at proper locations in x & rout */
  if((fr = fopen("resid.out","a")) == NULL) {
	printf("\007Error opening MAR coeff file!\n"); return -1;
  } 
  for(j=0;j<NTRLS;j++){
	for(k=0;k<NCHN;k++){
	  xtemp[k]=x[k]+j*WIN;
	  rtemp[k]=rout[k] + j*(WIN-MODORDER);  
	}
	MAR_residual(xtemp,rtemp,A,NCHN,MODORDER,WIN);

	/* compute mean and sum of squares for residuals of each channel */
	for(k=0;k<NCHN;k++){
	  mean[k] = getmean(rtemp[k],nresids);
	  ss[k] = getss(rtemp[k],mean[k],nresids);
	  /*	  fprintf(stderr,"%d %lf %lf\n", k, mean[k], ss[k]);*/
	}

	/* loop over channel pairs to get correlations */
	for(k=0;k<NCHN;k++){
	  for(l=0;l<NCHN;l++){	
	    if(l >= k){
		  cor(rtemp[k],mean[k],ss[k],rtemp[l],mean[l],ss[l],nresids,nlags,corfun);
	      for(lag=0;lag<nlags*2;lag++){
			ntot++;
			if(fabs(corfun[lag]) > sigthresh){
			  nsig++;
			}
	      }
	    }
	  }
	}

  } /* end of trial */
  prob = (double)nsig / (double)ntot;
  fprintf(fr,"%lf\n", prob);
  fclose(fr);

	/***  write MAR coefficients(including noise) to output files ***/
	/***  Each line of the output file corresponds to ONE MAR model at each 
	  given time instant  ***/

	if((fp = fopen(argv[2],"a")) == NULL) {
	  printf("\007Error opening MAR coeff file!\n"); return -1;
	} 
	/*   for ( i=0; i < 6; i++) *//* 5 order, 1st is identity; if 7, then new =0 */
   for ( i=0; i < MODORDER+1; i++) /* 5 order, 1st is identity; if 7, then new =0 */
	 for ( j=0; j < NCHN*NCHN; j++)
	   fprintf(fp,"%.3g  ",A[i][j]);
   fprintf(fp,"\n");
   fclose(fp);
   
  if((fp = fopen(argv[3],"a")) == NULL) {
	printf("\007Error opening MAR noise file!\n"); return -1;
  } 
   for ( i=0; i < NCHN*NCHN; i++)
	 fprintf(fp,"%.3g  ",Ve[i]);
   fprintf(fp,"\n");
   fclose(fp);
   
/*  if((fp = fopen(argv[6],"a")) == NULL) {
	printf("\007Error opening time file!\n"); return;
  } 
  fprintf(fp,"%d\n", t);
  fclose(fp);
 */  

  }   /* end of rec  */

  free(tildA);
  free(Ve);
  for(i=0;i<MAXORDER;i++)free(A[i]);
  free(n);

  for (i = 0; i < NCHN; i++)
    free(dat[i]);
  free(dat);

  for (i = 0; i < NCHN; i++)
    free(x[i]);
  free(x);

  for (j = 0; j < NCHN; j++){
	free(rout[j]);
  }
  free(rout);
  
  exit(0);

}







