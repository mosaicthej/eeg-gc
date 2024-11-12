#ifndef __MATMISC_H__
#define __MATMISC_H__

/* matmisc.c */
double* init_vec(int n);
double* init_mat(int m);
void vecmvech(double *a, double *b, double *C, int m);
void matmvec(double *A, double *b, double *c, int m);
void arraym(double *a, double *b, double *c, int n);
void matH(double *A, double *B, int m);
void arraycopy(double *A, double *B, int m);
void matmmat(double *A, double *B, double *C, int m);
void ltmatmmat(double *A, double *B, double *C, int m);
void matmutmat(double *A, double *B, double *C, int m);
void matmmatH(double *A, double *B, int m);
void arrayp(double *a, double *b, double *c, int n);
void zeromat(double *A, int m);
void Immat(double *A, double *B, int m);
void matms(double *A, double x,double *B, int m);

/* matmisc2.c */
void EEGerror(char *s);
void ludcmp(double *a, int n, int *indx, double *d);
void lubksb(double *a, int n, int *indx, double b[]);
void invmat(double *A, double *B, int m);
void detmat(double *A, double *x, int m);
void sqrmat(double *A, double *B, int m);
void revltmat(double *A, double *B, int m);
void zerovec(double *x, int n);

/* matmisc5.c */
void minmax(long nmax, int *x, int *xmin, int *xmax);
double cor(double *sigarray1, double mean1, double ss1, double *sigarray2,
double mean2, double ss2, int npoints, int nlags, double *corfun);
double getmean(double *array, int npoints);
double getss(double *array, double mean, int npoints);
double getcov(double *array1, double *array2, double mean1, double mean2, int npoints, int lag);

#endif /* __MATMISC_H__ */
