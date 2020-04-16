#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <complex.h>

typedef float  complex TCS;

#undef complex

#include "lapack-aux-float.h"

#define MACRO(B) do {B} while (0)
#define ERROR(CODE) MACRO(return CODE;)
#define REQUIRES(COND, CODE) MACRO(if(!(COND)) {ERROR(CODE);})

#define MIN(A,B) ((A)<(B)?(A):(B))
#define MAX(A,B) ((A)>(B)?(A):(B))

// #define DBGL

#ifdef DBGL
#define DEBUGMSG(M) printf("\nLAPACK "M"\n");
#else
#define DEBUGMSG(M)
#endif

#define OK return 0;

// #ifdef DBGL
// #define DEBUGMSG(M) printf("LAPACK Wrapper "M"\n: "); size_t t0 = time(NULL);
// #define OK MACRO(printf("%ld s\n",time(0)-t0); return 0;);
// #else
// #define DEBUGMSG(M)
// #define OK return 0;
// #endif


#define INFOMAT(M) printf("%dx%d %d:%d\n",M##r,M##c,M##Xr,M##Xc);

#define TRACEMAT(M) {int q; printf(" %d x %d: ",M##r,M##c); \
                     for(q=0;q<M##r*M##c;q++) printf("%.1f ",M##p[q]); printf("\n");}

#define CHECK(RES,CODE) MACRO(if(RES) return CODE;)

#define BAD_SIZE 2000
#define BAD_CODE 2001
#define MEM      2002
#define BAD_FILE 2003
#define SINGULAR 2004
#define NOCONVER 2005
#define NODEFPOS 2006
#define NOSPRTD  2007

////////////////////////////////////////////////////////////////////////////////
void asm_finit() {
#ifdef i386

//  asm("finit");

    static unsigned char buf[108];
    asm("FSAVE %0":"=m" (buf));

    #if FPUDEBUG
    if(buf[8]!=255 || buf[9]!=255) {  // print warning in red
        printf("%c[;31mWarning: FPU TAG = %x %x\%c[0m\n",0x1B,buf[8],buf[9],0x1B);
    }
    #endif

    #if NANDEBUG
    asm("FRSTOR %0":"=m" (buf));
    #endif

#endif
}

#if NANDEBUG

#define CHECKNANR(M,msg)                     \
{ int k;                                     \
for(k=0; k<(M##r * M##c); k++) {             \
    if(M##p[k] != M##p[k]) {                 \
        printf(msg);                         \
        TRACEMAT(M)                          \
        /*exit(1);*/                         \
    }                                        \
}                                            \
}

#define CHECKNANC(M,msg)                     \
{ int k;                                     \
for(k=0; k<(M##r * M##c); k++) {             \
    if(  M##p[k].r != M##p[k].r              \
      || M##p[k].i != M##p[k].i) {           \
        printf(msg);                         \
        /*exit(1);*/                         \
    }                                        \
}                                            \
}

#else
#define CHECKNANC(M,msg)
#define CHECKNANR(M,msg)
#endif


////////////////////////////////////////////////////////////////////////////////
//////////////////// real svd ///////////////////////////////////////////////////

int sgesvd_(char *jobu, char *jobvt, integer *m, integer *n,
	real *a, integer *lda, real *s, real *u, integer *
	ldu, real *vt, integer *ldvt, real *work, integer *lwork,
	integer *info);

int svd_l_S(OFMAT(a),OFMAT(u), FVEC(s),OFMAT(v)) {
    integer m = ar;
    integer n = ac;
    integer q = MIN(m,n);
    REQUIRES(sn==q,BAD_SIZE);
    REQUIRES(up==NULL || (ur==m && (uc==m || uc==q)),BAD_SIZE);
    char* jobu  = "A";
    if (up==NULL) {
        jobu = "N";
    } else {
        if (uc==q) {
            jobu = "S";
        }
    }
    REQUIRES(vp==NULL || (vc==n && (vr==n || vr==q)),BAD_SIZE);
    char* jobvt  = "A";
    integer ldvt = n;
    if (vp==NULL) {
        jobvt = "N";
    } else {
        if (vr==q) {
            jobvt = "S";
            ldvt = q;
        }
    }
    DEBUGMSG("svd_l_R");
    integer lwork = -1;
    integer res;
    // ask for optimal lwork
    float ans;
    sgesvd_ (jobu,jobvt,
             &m,&n,ap,&m,
             sp,
             up,&m,
             vp,&ldvt,
             &ans, &lwork,
             &res);
    lwork = ceil(ans);
    float * work = (float*)malloc(lwork*sizeof(float));
    CHECK(!work,MEM);
    sgesvd_ (jobu,jobvt,
             &m,&n,ap,&m,
             sp,
             up,&m,
             vp,&ldvt,
             work, &lwork,
             &res);
    CHECK(res,res);
    free(work);
    OK
}

// (alternative version)

int sgesdd_(char *jobz, integer *m, integer *n, real *
	a, integer *lda, real *s, real *u, integer *ldu,
	real *vt, integer *ldvt, real *work, integer *lwork,
	integer *iwork, integer *info);

int svd_l_Rdd(OFMAT(a),OFMAT(u), FVEC(s),OFMAT(v)) {
    integer m = ar;
    integer n = ac;
    integer q = MIN(m,n);
    REQUIRES(sn==q,BAD_SIZE);
    REQUIRES((up == NULL && vp == NULL)
             || (ur==m && vc==n
                &&   ((uc == q && vr == q)
                   || (uc == m && vc==n))),BAD_SIZE);
    char* jobz  = "A";
    integer ldvt = n;
    if (up==NULL) {
        jobz = "N";
    } else {
        if (uc==q && vr == q) {
            jobz = "S";
            ldvt = q;
        }
    }
    DEBUGMSG("svd_l_Rdd");
    integer* iwk = (integer*) malloc(8*q*sizeof(integer));
    CHECK(!iwk,MEM);
    integer lwk = -1;
    integer res;
    // ask for optimal lwk
    float ans;
    sgesdd_ (jobz,&m,&n,ap,&m,sp,up,&m,vp,&ldvt,&ans,&lwk,iwk,&res);
    lwk = ans;
    float * workv = (float*)malloc(lwk*sizeof(float));
    CHECK(!workv,MEM);
    sgesdd_ (jobz,&m,&n,ap,&m,sp,up,&m,vp,&ldvt,workv,&lwk,iwk,&res);
    CHECK(res,res);
    free(iwk);
    free(workv);
    OK
}

//////////////////// complex svd ////////////////////////////////////

int cgesvd_(char *jobu, char *jobvt, integer *m, integer *n,
    complex *a, integer *lda, real *s, complex *u,
    integer *ldu, complex *vt, integer *ldvt, complex *work,
    integer *lwork, real *rwork, integer *info);

int svd_l_C(OQMAT(a),OQMAT(u), FVEC(s),OQMAT(v)) {
    integer m = ar;
    integer n = ac;
    integer q = MIN(m,n);
    REQUIRES(sn==q,BAD_SIZE);
    REQUIRES(up==NULL || (ur==m && (uc==m || uc==q)),BAD_SIZE);
    char* jobu  = "A";
    if (up==NULL) {
        jobu = "N";
    } else {
        if (uc==q) {
            jobu = "S";
        }
    }
    REQUIRES(vp==NULL || (vc==n && (vr==n || vr==q)),BAD_SIZE);
    char* jobvt  = "A";
    integer ldvt = n;
    if (vp==NULL) {
        jobvt = "N";
    } else {
        if (vr==q) {
            jobvt = "S";
            ldvt = q;
        }
    }DEBUGMSG("svd_l_C");

    float *rwork = (float*) malloc(5*q*sizeof(float));
    CHECK(!rwork,MEM);
    integer lwork = -1;
    integer res;
    // ask for optimal lwork
    complex ans;
    cgesvd_ (jobu,jobvt,
             &m,&n,ap,&m,
             sp,
             up,&m,
             vp,&ldvt,
             &ans, &lwork,
             rwork,
             &res);
    lwork = ceil(ans.r);
    complex * work = (complex*)malloc(lwork*sizeof(complex));
    CHECK(!work,MEM);
    cgesvd_ (jobu,jobvt,
             &m,&n,ap,&m,
             sp,
             up,&m,
             vp,&ldvt,
             work, &lwork,
             rwork,
             &res);
    CHECK(res,res);
    free(work);
    free(rwork);
    OK
}

int cgesdd_ (char *jobz, integer *m, integer *n,
    complex *a, integer *lda, real *s, complex *u,
    integer *ldu, complex *vt, integer *ldvt, complex *work,
    integer *lwork, real *rwork, integer* iwork, integer *info);

int svd_l_Cdd(OQMAT(a),OQMAT(u), FVEC(s),OQMAT(v)) {
    integer m = ar;
    integer n = ac;
    integer q = MIN(m,n);
    REQUIRES(sn==q,BAD_SIZE);
    REQUIRES((up == NULL && vp == NULL)
             || (ur==m && vc==n
                &&   ((uc == q && vr == q)
                   || (uc == m && vc==n))),BAD_SIZE);
    char* jobz  = "A";
    integer ldvt = n;
    if (up==NULL) {
        jobz = "N";
    } else {
        if (uc==q && vr == q) {
            jobz = "S";
            ldvt = q;
        }
    }
    DEBUGMSG("svd_l_Cdd");
    integer* iwk = (integer*) malloc(8*q*sizeof(integer));
    CHECK(!iwk,MEM);
    int lrwk;
    if (0 && *jobz == 'N') {
        lrwk = 5*q; // does not work, crash at free below
    } else {
        lrwk = 5*q*q + 7*q;
    }
    float *rwk = (float*)malloc(lrwk*sizeof(float));;
    CHECK(!rwk,MEM);
    integer lwk = -1;
    integer res;
    // ask for optimal lwk
    complex ans;
    cgesdd_ (jobz,&m,&n,ap,&m,sp,up,&m,vp,&ldvt,&ans,&lwk,rwk,iwk,&res);
    lwk = ans.r;
    complex * workv = (complex*)malloc(lwk*sizeof(complex));
    CHECK(!workv,MEM);
    cgesdd_ (jobz,&m,&n,ap,&m,sp,up,&m,vp,&ldvt,workv,&lwk,rwk,iwk,&res);
    CHECK(res,res);
    free(workv);
    free(rwk);
    free(iwk);
    OK
}

//////////////////// general complex eigensystem ////////////

int cgeev_(char *jobvl, char *jobvr, integer *n,
	complex *a, integer *lda, complex *w, complex *vl,
	integer *ldvl, complex *vr, integer *ldvr, complex *work,
	integer *lwork, real *rwork, integer *info);

int eig_l_C(OQMAT(a), OQMAT(u), QVEC(s),OQMAT(v)) {
    integer n = ar;
    REQUIRES(ac==n && sn==n, BAD_SIZE);
    REQUIRES(up==NULL || (ur==n && uc==n), BAD_SIZE);
    char jobvl = up==NULL?'N':'V';
    REQUIRES(vp==NULL || (vr==n && vc==n), BAD_SIZE);
    char jobvr = vp==NULL?'N':'V';
    DEBUGMSG("eig_l_C");
    float *rwork = (float*) malloc(2*n*sizeof(float));
    CHECK(!rwork,MEM);
    integer lwork = -1;
    integer res;
    // ask for optimal lwork
    complex ans;
    cgeev_  (&jobvl,&jobvr,
             &n,ap,&n,
             sp,
             up,&n,
             vp,&n,
             &ans, &lwork,
             rwork,
             &res);
    lwork = ceil(ans.r);
    complex * work = (complex*)malloc(lwork*sizeof(complex));
    CHECK(!work,MEM);
    cgeev_  (&jobvl,&jobvr,
             &n,ap,&n,
             sp,
             up,&n,
             vp,&n,
             work, &lwork,
             rwork,
             &res);
    CHECK(res,res);
    free(work);
    free(rwork);
    OK
}


//////////////////// general real eigensystem ////////////

int sgeev_(char *jobvl, char *jobvr, integer *n, real *
	a, integer *lda, real *wr, real *wi, real *vl,
	integer *ldvl, real *vr, integer *ldvr, real *work,
	integer *lwork, integer *info);

int eig_l_R(OFMAT(a),OFMAT(u), QVEC(s),OFMAT(v)) {
    integer n = ar;
    REQUIRES(ac==n && sn==n, BAD_SIZE);
    REQUIRES(up==NULL || (ur==n && uc==n), BAD_SIZE);
    char jobvl = up==NULL?'N':'V';
    REQUIRES(vp==NULL || (vr==n && vc==n), BAD_SIZE);
    char jobvr = vp==NULL?'N':'V';
    DEBUGMSG("eig_l_R");
    integer lwork = -1;
    integer res;
    // ask for optimal lwork
    float ans;
    sgeev_  (&jobvl,&jobvr,
             &n,ap,&n,
             (float*)sp, (float*)sp+n,
             up,&n,
             vp,&n,
             &ans, &lwork,
             &res);
    lwork = ceil(ans);
    float * work = (float*)malloc(lwork*sizeof(float));
    CHECK(!work,MEM);
    sgeev_  (&jobvl,&jobvr,
             &n,ap,&n,
             (float*)sp, (float*)sp+n,
             up,&n,
             vp,&n,
             work, &lwork,
             &res);
    CHECK(res,res);
    free(work);
    OK
}

//////////////////// generalized real eigensystem ////////////

int sggev_(char *jobvl, char *jobvr, integer *n,
    real *a, integer *lda, real *b, integer *ldb,
    real *alphar, real *alphai, real *beta,
    real *vl, integer *ldvl, real *vr, integer *ldvr,
    real *work,
	integer *lwork, integer *info);

int eig_l_G(OFMAT(a), OFMAT(b), QVEC(alpha), FVEC(beta), OFMAT(vl), OFMAT(vr)) {
    integer n = ar;
    REQUIRES(ac == n && br == n && bc == n && alphan == n && betan == n, BAD_SIZE);
    REQUIRES(vlp==NULL || (vlr==n && vlc==n), BAD_SIZE);
    char jobvl = vlp==NULL?'N':'V';
    REQUIRES(vrp==NULL || (vrr==n && vrc==n), BAD_SIZE);
    char jobvr = vrp==NULL?'N':'V';
    DEBUGMSG("eig_l_G");
    integer lwork = -1;
    integer res;
    // ask for optimal lwork
    float ans;
    sggev_  (&jobvl,&jobvr,
             &n,
             ap,&n,bp,&n,
             (float*)alphap, (float*)alphap+n, betap,
             vlp, &n, vrp, &n,
             &ans, &lwork,
             &res);
    lwork = ceil(ans);
    float * work = (float*)malloc(lwork*sizeof(float));
    CHECK(!work,MEM);
    sggev_  (&jobvl,&jobvr,
             &n,
             ap,&n,bp,&n,
             (float*)alphap, (float*)alphap+n, betap,
             vlp, &n, vrp, &n,
             work, &lwork,
             &res);
    CHECK(res,res);
    free(work);
    OK
}

//////////////////// generalized complex eigensystem ////////////

int cggev_(char *jobvl, char *jobvr, integer *n,
    complex *a, integer *lda, complex *b, integer *ldb,
    complex *alphar, complex *beta,
    complex *vl, integer *ldvl, complex *vr, integer *ldvr,
    complex *work, integer *lwork,
    real *rwork, integer *info);

int eig_l_GC(OQMAT(a), OQMAT(b), QVEC(alpha), QVEC(beta), OQMAT(vl), OQMAT(vr)) {
    integer n = ar;
    REQUIRES(ac == n && br == n && bc == n && alphan == n && betan == n, BAD_SIZE);
    REQUIRES(vlp==NULL || (vlr==n && vlc==n), BAD_SIZE);
    char jobvl = vlp==NULL?'N':'V';
    REQUIRES(vrp==NULL || (vrr==n && vrc==n), BAD_SIZE);
    char jobvr = vrp==NULL?'N':'V';
    DEBUGMSG("eig_l_GC");
    float *rwork = (float*) malloc(8*n*sizeof(float));
    CHECK(!rwork,MEM);
    integer lwork = -1;
    integer res;
    // ask for optimal lwork
    complex ans;
    cggev_  (&jobvl,&jobvr,
             &n,
             ap,&n,bp,&n,
             alphap, betap,
             vlp, &n, vrp, &n,
             &ans, &lwork,
             rwork, &res);
    lwork = ceil(ans.r);
    complex * work = (complex*)malloc(lwork*sizeof(complex));
    CHECK(!work,MEM);
    cggev_  (&jobvl,&jobvr,
             &n,
             ap,&n,bp,&n,
             alphap, betap,
             vlp, &n, vrp, &n,
             work, &lwork,
             rwork, &res);
    CHECK(res,res);
    free(work);
    OK
}

//////////////////// symmetric real eigensystem ////////////

int ssyev_(char *jobz, char *uplo, integer *n, real *a,
	integer *lda, real *w, real *work, integer *lwork,
	integer *info);

int eig_l_S(int wantV,FVEC(s),OFMAT(v)) {
    integer n = sn;
    REQUIRES(vr==n && vc==n, BAD_SIZE);
    char jobz = wantV?'V':'N';
    DEBUGMSG("eig_l_S");
    integer lwork = -1;
    char uplo = 'U';
    integer res;
    // ask for optimal lwork
    float ans;
    ssyev_  (&jobz,&uplo,
             &n,vp,&n,
             sp,
             &ans, &lwork,
             &res);
    lwork = ceil(ans);
    float * work = (float*)malloc(lwork*sizeof(float));
    CHECK(!work,MEM);
    ssyev_  (&jobz,&uplo,
             &n,vp,&n,
             sp,
             work, &lwork,
             &res);
    CHECK(res,res);
    free(work);
    OK
}

//////////////////// hermitian complex eigensystem ////////////

int cheev_(char *jobz, char *uplo, integer *n, complex
	*a, integer *lda, real *w, complex *work, integer *lwork,
	real *rwork, integer *info);

int eig_l_H(int wantV,FVEC(s),OQMAT(v)) {
    integer n = sn;
    REQUIRES(vr==n && vc==n, BAD_SIZE);
    char jobz = wantV?'V':'N';
    DEBUGMSG("eig_l_H");
    float *rwork = (float*) malloc((3*n-2)*sizeof(float));
    CHECK(!rwork,MEM);
    integer lwork = -1;
    char uplo = 'U';
    integer res;
    // ask for optimal lwork
    complex ans;
    cheev_  (&jobz,&uplo,
             &n,vp,&n,
             sp,
             &ans, &lwork,
             rwork,
             &res);
    lwork = ceil(ans.r);
    complex * work = (complex*)malloc(lwork*sizeof(complex));
    CHECK(!work,MEM);
    cheev_  (&jobz,&uplo,
             &n,vp,&n,
             sp,
             work, &lwork,
             rwork,
             &res);
    CHECK(res,res);
    free(work);
    free(rwork);
    OK
}

//////////////////// general real linear system ////////////

int sgesv_(integer *n, integer *nrhs, real *a, integer
	*lda, integer *ipiv, real *b, integer *ldb, integer *info);

int linearSolveR_l(OFMAT(a),OFMAT(b)) {
    integer n = ar;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("linearSolveR_l");
    integer * ipiv = (integer*)malloc(n*sizeof(integer));
    integer res;
    sgesv_  (&n,&nhrs,
             ap, &n,
             ipiv,
             bp, &n,
             &res);
    if(res>0) {
        return SINGULAR;
    }
    CHECK(res,res);
    free(ipiv);
    OK
}

//////////////////// general complex linear system ////////////

int cgesv_(integer *n, integer *nrhs, complex *a,
	integer *lda, integer *ipiv, complex *b, integer *ldb, integer *
	info);

int linearSolveC_l(OQMAT(a),OQMAT(b)) {
    integer n = ar;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("linearSolveC_l");
    integer * ipiv = (integer*)malloc(n*sizeof(integer));
    integer res;
    cgesv_  (&n,&nhrs,
             ap, &n,
             ipiv,
             bp, &n,
             &res);
    if(res>0) {
        return SINGULAR;
    }
    CHECK(res,res);
    free(ipiv);
    OK
}

//////// symmetric positive definite real linear system using Cholesky ////////////

int spotrs_(char *uplo, integer *n, integer *nrhs,
	real *a, integer *lda, real *b, integer *ldb, integer *
	info);

int cholSolveR_l(KOFMAT(a),OFMAT(b)) {
    integer n = ar;
    integer lda = aXc;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("cholSolveR_l");
    integer res;
    spotrs_ ("U",
             &n,&nhrs,
             (float*)ap, &lda,
             bp, &n,
             &res);
    CHECK(res,res);
    OK
}

//////// Hermitian positive definite real linear system using Cholesky ////////////

int cpotrs_(char *uplo, integer *n, integer *nrhs,
	complex *a, integer *lda, complex *b, integer *ldb,
	integer *info);

int cholSolveC_l(KOQMAT(a),OQMAT(b)) {
    integer n = ar;
    integer lda = aXc;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("cholSolveC_l");
    integer res;
    cpotrs_  ("U",
             &n,&nhrs,
             (complex*)ap, &lda,
             bp, &n,
             &res);
    CHECK(res,res);
    OK
}

//////// triangular real linear system ////////////

int strtrs_(char *uplo, char *trans, char *diag, integer *n, integer *nrhs,
	real *a, integer *lda, real *b, integer *ldb, integer *
	info);

int triSolveR_l_u(KOFMAT(a),OFMAT(b)) {
    integer n = ar;
    integer lda = aXc;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("triSolveR_l_u");
    integer res;
    strtrs_ ("U",
             "N",
             "N",
             &n,&nhrs,
             (float*)ap, &lda,
             bp, &n,
             &res);
    CHECK(res,res);
    OK
}

int triSolveR_l_l(KOFMAT(a),OFMAT(b)) {
    integer n = ar;
    integer lda = aXc;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("triSolveR_l_l");
    integer res;
    strtrs_ ("L",
             "N",
             "N",
             &n,&nhrs,
             (float*)ap, &lda,
             bp, &n,
             &res);
    CHECK(res,res);
    OK
}

//////// triangular complex linear system ////////////

int ctrtrs_(char *uplo, char *trans, char *diag, integer *n, integer *nrhs,
	complex *a, integer *lda, complex *b, integer *ldb,
	integer *info);

int triSolveC_l_u(KOQMAT(a),OQMAT(b)) {
    integer n = ar;
    integer lda = aXc;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("triSolveC_l_u");
    integer res;
    ctrtrs_ ("U",
             "N",
             "N",
             &n,&nhrs,
             (complex*)ap, &lda,
             bp, &n,
             &res);
    CHECK(res,res);
    OK
}

int triSolveC_l_l(KOQMAT(a),OQMAT(b)) {
    integer n = ar;
    integer lda = aXc;
    integer nhrs = bc;
    REQUIRES(n>=1 && ar==ac && ar==br,BAD_SIZE);
    DEBUGMSG("triSolveC_l_u");
    integer res;
    ctrtrs_ ("L",
             "N",
             "N",
             &n,&nhrs,
             (complex*)ap, &lda,
             bp, &n,
             &res);
    CHECK(res,res);
    OK
}

//////// tridiagonal real linear system ////////////

int sgttrf_(integer *n,
            real *dl, real *d, real *du, real *du2,
            integer *ipiv,
            integer *info);

int sgttrs_(char *trans, integer *n, integer *nrhs,
            real *dl, real *d, real *du, real *du2,
            integer *ipiv, real *b, integer *ldb,
            integer *info);

int triDiagSolveR_l(FVEC(dl), FVEC(d), FVEC(du), OFMAT(b)) {
    integer n = dn;
    integer nhrs = bc;
    REQUIRES(n >= 1 && dln == dn - 1 && dun == dn - 1 && br == n, BAD_SIZE);
    DEBUGMSG("triDiagSolveR_l");
    integer res;
    integer* ipiv = (integer*)malloc(n*sizeof(integer));
    float* du2  = (float*)malloc((n - 2)*sizeof(float));
    sgttrf_ (&n,
             dlp, dp, dup, du2,
             ipiv,
             &res);
    CHECK(res,res);
    sgttrs_ ("N",
             &n,&nhrs,
             dlp, dp, dup, du2,
             ipiv, bp, &n,
             &res);
    CHECK(res,res);
    free(ipiv);
    free(du2);
    OK
}

//////// tridiagonal complex linear system ////////////

int cgttrf_(integer *n,
            complex *dl, complex *d, complex *du, complex *du2,
            integer *ipiv,
            integer *info);

int cgttrs_(char *trans, integer *n, integer *nrhs,
            complex *dl, complex *d, complex *du, complex *du2,
            integer *ipiv, complex *b, integer *ldb,
            integer *info);

int triDiagSolveC_l(QVEC(dl), QVEC(d), QVEC(du), OQMAT(b)) {
    integer n = dn;
    integer nhrs = bc;
    REQUIRES(n >= 1 && dln == dn - 1 && dun == dn - 1 && br == n, BAD_SIZE);
    DEBUGMSG("triDiagSolveC_l");
    integer res;
    integer* ipiv = (integer*)malloc(n*sizeof(integer));
    complex* du2 = (complex*)malloc((n - 2)*sizeof(complex));
    cgttrf_ (&n,
             dlp, dp, dup, du2,
             ipiv,
             &res);
    CHECK(res,res);
    cgttrs_ ("N",
             &n,&nhrs,
             dlp, dp, dup, du2,
             ipiv, bp, &n,
             &res);
    CHECK(res,res);
    free(ipiv);
    free(du2);
    OK
}

//////////////////// least squares real linear system ////////////

int sgels_(char *trans, integer *m, integer *n, integer *
	nrhs, real *a, integer *lda, real *b, integer *ldb,
	real *work, integer *lwork, integer *info);

int linearSolveLSR_l(OFMAT(a),OFMAT(b)) {
    integer m = ar;
    integer n = ac;
    integer nrhs = bc;
    integer ldb = bXc;
    REQUIRES(m>=1 && n>=1 && br==MAX(m,n), BAD_SIZE);
    DEBUGMSG("linearSolveLSR_l");
    integer res;
    integer lwork = -1;
    float ans;
    sgels_  ("N",&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             &ans,&lwork,
             &res);
    lwork = ceil(ans);
    float * work = (float*)malloc(lwork*sizeof(float));
    sgels_  ("N",&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             work,&lwork,
             &res);
    if(res>0) {
        return SINGULAR;
    }
    CHECK(res,res);
    free(work);
    OK
}

//////////////////// least squares complex linear system ////////////

int cgels_(char *trans, integer *m, integer *n, integer *
	nrhs, complex *a, integer *lda, complex *b, integer *ldb,
	complex *work, integer *lwork, integer *info);

int linearSolveLSC_l(OQMAT(a),OQMAT(b)) {
    integer m = ar;
    integer n = ac;
    integer nrhs = bc;
    integer ldb = bXc;
    REQUIRES(m>=1 && n>=1 && br==MAX(m,n), BAD_SIZE);
    DEBUGMSG("linearSolveLSC_l");
    integer res;
    integer lwork = -1;
    complex ans;
    cgels_  ("N",&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             &ans,&lwork,
             &res);
    lwork = ceil(ans.r);
    complex * work = (complex*)malloc(lwork*sizeof(complex));
    cgels_  ("N",&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             work,&lwork,
             &res);
    if(res>0) {
        return SINGULAR;
    }
    CHECK(res,res);
    free(work);
    OK
}

//////////////////// least squares real linear system using SVD ////////////

int sgelss_(integer *m, integer *n, integer *nrhs,
	real *a, integer *lda, real *b, integer *ldb, real *
	s, real *rcond, integer *rank, real *work, integer *lwork,
	integer *info);

int linearSolveSVDR_l(float rcond,OFMAT(a),OFMAT(b)) {
    integer m = ar;
    integer n = ac;
    integer nrhs = bc;
    integer ldb = bXc;
    REQUIRES(m>=1 && n>=1 && br==MAX(m,n), BAD_SIZE);
    DEBUGMSG("linearSolveSVDR_l");
    float*S = (float*)malloc(MIN(m,n)*sizeof(float));
    integer res;
    integer lwork = -1;
    integer rank;
    float ans;
    sgelss_  (&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             S,
             &rcond,&rank,
             &ans,&lwork,
             &res);
    lwork = ceil(ans);
    float * work = (float*)malloc(lwork*sizeof(float));
    sgelss_  (&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             S,
             &rcond,&rank,
             work,&lwork,
             &res);
    if(res>0) {
        return NOCONVER;
    }
    CHECK(res,res);
    free(work);
    free(S);
    OK
}

//////////////////// least squares complex linear system using SVD ////////////

int cgelss_(integer *m, integer *n, integer *nhrs,
    complex *a, integer *lda, complex *b, integer *ldb, real *s,
    real *rcond, integer* rank,
    complex *work, integer* lwork, real* rwork,
    integer *info);

int linearSolveSVDC_l(float rcond, OQMAT(a),OQMAT(b)) {
    integer m = ar;
    integer n = ac;
    integer nrhs = bc;
    integer ldb = bXc;
    REQUIRES(m>=1 && n>=1 && br==MAX(m,n), BAD_SIZE);
    DEBUGMSG("linearSolveSVDC_l");
    float*S = (float*)malloc(MIN(m,n)*sizeof(float));
    float*RWORK = (float*)malloc(5*MIN(m,n)*sizeof(float));
    integer res;
    integer lwork = -1;
    integer rank;
    complex ans;
    cgelss_  (&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             S,
             &rcond,&rank,
             &ans,&lwork,
             RWORK,
             &res);
    lwork = ceil(ans.r);
    complex * work = (complex*)malloc(lwork*sizeof(complex));
    cgelss_  (&m,&n,&nrhs,
             ap,&m,
             bp,&ldb,
             S,
             &rcond,&rank,
             work,&lwork,
             RWORK,
             &res);
    if(res>0) {
        return NOCONVER;
    }
    CHECK(res,res);
    free(work);
    free(RWORK);
    free(S);
    OK
}

//////////////////// Cholesky factorization /////////////////////////

int cpotrf_(char *uplo, integer *n, complex *a, integer *lda, integer *info);

int chol_l_H(OQMAT(l)) {
    integer n = lr;
    REQUIRES(n>=1 && lc == n,BAD_SIZE);
    DEBUGMSG("chol_l_H");
    char uplo = 'U';
    integer res;
    cpotrf_ (&uplo,&n,lp,&n,&res);
    CHECK(res>0,NODEFPOS);
    CHECK(res,res);
    complex zero = {0.,0.};
    int r,c;
    for (r=0; r<lr; r++) {
        for(c=0; c<r; c++) {
            AT(l,r,c) = zero;
        }
    }
    OK
}


int spotrf_(char *uplo, integer *n, real *a, integer * lda, integer *info);

int chol_l_S(OFMAT(l)) {
    integer n = lr;
    REQUIRES(n>=1 && lc == n,BAD_SIZE);
    DEBUGMSG("chol_l_S");
    char uplo = 'U';
    integer res;
    spotrf_ (&uplo,&n,lp,&n,&res);
    CHECK(res>0,NODEFPOS);
    CHECK(res,res);
    int r,c;
    for (r=0; r<lr; r++) {
        for(c=0; c<r; c++) {
            AT(l,r,c) = 0.;
        }
    }
    OK
}

//////////////////// QR factorization /////////////////////////

int sgeqr2_(integer *m, integer *n, real *a, integer *
	lda, real *tau, real *work, integer *info);

int qr_l_R(FVEC(tau), OFMAT(r)) {
    integer m = rr;
    integer n = rc;
    integer mn = MIN(m,n);
    REQUIRES(m>=1 && n >=1 && taun == mn, BAD_SIZE);
    DEBUGMSG("qr_l_R");
    float *WORK = (float*)malloc(n*sizeof(float));
    CHECK(!WORK,MEM);
    integer res;
    sgeqr2_ (&m,&n,rp,&m,taup,WORK,&res);
    CHECK(res,res);
    free(WORK);
    OK
}

int cgeqr2_(integer *m, integer *n, complex *a,
	integer *lda, complex *tau, complex *work, integer *info);

int qr_l_C(QVEC(tau), OQMAT(r)) {
    integer m = rr;
    integer n = rc;
    integer mn = MIN(m,n);
    REQUIRES(m>=1 && n >=1 && taun == mn, BAD_SIZE);
    DEBUGMSG("qr_l_C");
    complex *WORK = (complex*)malloc(n*sizeof(complex));
    CHECK(!WORK,MEM);
    integer res;
    cgeqr2_ (&m,&n,rp,&m,taup,WORK,&res);
    CHECK(res,res);
    free(WORK);
    OK
}

int sorgqr_(integer *m, integer *n, integer *k, real *
	a, integer *lda, real *tau, real *work, integer *lwork,
	integer *info);

int c_dorgqr(KFVEC(tau), OFMAT(r)) {
    integer m = rr;
    integer n = MIN(rc,rr);
    integer k = taun;
    DEBUGMSG("c_dorgqr");
    integer lwork = 8*n; // FIXME
    float *WORK = (float*)malloc(lwork*sizeof(float));
    CHECK(!WORK,MEM);
    integer res;
    sorgqr_ (&m,&n,&k,rp,&m,(float*)taup,WORK,&lwork,&res);
    CHECK(res,res);
    free(WORK);
    OK
}

int cungqr_(integer *m, integer *n, integer *k,
	complex *a, integer *lda, complex *tau, complex *
	work, integer *lwork, integer *info);

int c_zungqr(KQVEC(tau), OQMAT(r)) {
    integer m = rr;
    integer n = MIN(rc,rr);
    integer k = taun;
    DEBUGMSG("z_ungqr");
    integer lwork = 8*n; // FIXME
    complex *WORK = (complex*)malloc(lwork*sizeof(complex));
    CHECK(!WORK,MEM);
    integer res;
    cungqr_ (&m,&n,&k,rp,&m,(complex*)taup,WORK,&lwork,&res);
    CHECK(res,res);
    free(WORK);
    OK
}


//////////////////// Hessenberg factorization /////////////////////////

int sgehrd_(integer *n, integer *ilo, integer *ihi,
	real *a, integer *lda, real *tau, real *work,
	integer *lwork, integer *info);

int hess_l_R(FVEC(tau), OFMAT(r)) {
    integer m = rr;
    integer n = rc;
    integer mn = MIN(m,n);
    REQUIRES(m>=1 && n == m && taun == mn-1, BAD_SIZE);
    DEBUGMSG("hess_l_R");
    integer lwork = 5*n; // FIXME
    float *WORK = (float*)malloc(lwork*sizeof(float));
    CHECK(!WORK,MEM);
    integer res;
    integer one = 1;
    sgehrd_ (&n,&one,&n,rp,&n,taup,WORK,&lwork,&res);
    CHECK(res,res);
    free(WORK);
    OK
}


int cgehrd_(integer *n, integer *ilo, integer *ihi,
	complex *a, integer *lda, complex *tau, complex *
	work, integer *lwork, integer *info);

int hess_l_C(QVEC(tau), OQMAT(r)) {
    integer m = rr;
    integer n = rc;
    integer mn = MIN(m,n);
    REQUIRES(m>=1 && n == m && taun == mn-1, BAD_SIZE);
    DEBUGMSG("hess_l_C");
    integer lwork = 5*n; // FIXME
    complex *WORK = (complex*)malloc(lwork*sizeof(complex));
    CHECK(!WORK,MEM);
    integer res;
    integer one = 1;
    cgehrd_ (&n,&one,&n,rp,&n,taup,WORK,&lwork,&res);
    CHECK(res,res);
    free(WORK);
    OK
}

//////////////////// Schur factorization /////////////////////////

int sgees_(char *jobvs, char *sort, L_fp select, integer *n,
	real *a, integer *lda, integer *sdim, real *wr,
	real *wi, real *vs, integer *ldvs, real *work,
	integer *lwork, logical *bwork, integer *info);

int schur_l_R(OFMAT(u), OFMAT(s)) {
    integer m = sr;
    integer n = sc;
    REQUIRES(m>=1 && n==m && ur==n && uc==n, BAD_SIZE);
    DEBUGMSG("schur_l_R");
    integer lwork = 6*n; // FIXME
    float *WORK = (float*)malloc(lwork*sizeof(float));
    float *WR = (float*)malloc(n*sizeof(float));
    float *WI = (float*)malloc(n*sizeof(float));
    // WR and WI not really required in this call
    logical *BWORK = (logical*)malloc(n*sizeof(logical));
    integer res;
    integer sdim;
    sgees_ ("V","N",NULL,&n,sp,&n,&sdim,WR,WI,up,&n,WORK,&lwork,BWORK,&res);
    if(res>0) {
        return NOCONVER;
    }
    CHECK(res,res);
    free(WR);
    free(WI);
    free(BWORK);
    free(WORK);
    OK
}


int cgees_(char *jobvs, char *sort, L_fp select, integer *n,
	complex *a, integer *lda, integer *sdim, complex *w,
	complex *vs, integer *ldvs, complex *work, integer *lwork,
	real *rwork, logical *bwork, integer *info);

int schur_l_C(OQMAT(u), OQMAT(s)) {
    integer m = sr;
    integer n = sc;
    REQUIRES(m>=1 && n==m && ur==n && uc==n, BAD_SIZE);
    DEBUGMSG("schur_l_C");
    integer lwork = 6*n; // FIXME
    complex *WORK = (complex*)malloc(lwork*sizeof(complex));
    complex *W = (complex*)malloc(n*sizeof(complex));
    // W not really required in this call
    logical *BWORK = (logical*)malloc(n*sizeof(logical));
    float *RWORK = (float*)malloc(n*sizeof(float));
    integer res;
    integer sdim;
    cgees_ ("V","N",NULL,&n,sp,&n,&sdim,W,
                            up,&n,
                            WORK,&lwork,RWORK,BWORK,&res);
    if(res>0) {
        return NOCONVER;
    }
    CHECK(res,res);
    free(W);
    free(BWORK);
    free(WORK);
    OK
}

//////////////////// LU factorization /////////////////////////

int sgetrf_(integer *m, integer *n, real *a, integer *
	lda, integer *ipiv, integer *info);

int lu_l_R(FVEC(ipiv), OFMAT(r)) {
    integer m = rr;
    integer n = rc;
    integer mn = MIN(m,n);
    REQUIRES(m>=1 && n >=1 && ipivn == mn, BAD_SIZE);
    DEBUGMSG("lu_l_R");
    integer* auxipiv = (integer*)malloc(mn*sizeof(integer));
    integer res;
    sgetrf_ (&m,&n,rp,&m,auxipiv,&res);
    if(res>0) {
        res = 0; // FIXME
    }
    CHECK(res,res);
    int k;
    for (k=0; k<mn; k++) {
        ipivp[k] = auxipiv[k];
    }
    free(auxipiv);
    OK
}


int cgetrf_(integer *m, integer *n, complex *a,
	integer *lda, integer *ipiv, integer *info);

int lu_l_C(FVEC(ipiv), OQMAT(r)) {
    integer m = rr;
    integer n = rc;
    integer mn = MIN(m,n);
    REQUIRES(m>=1 && n >=1 && ipivn == mn, BAD_SIZE);
    DEBUGMSG("lu_l_C");
    integer* auxipiv = (integer*)malloc(mn*sizeof(integer));
    integer res;
    cgetrf_ (&m,&n,rp,&m,auxipiv,&res);
    if(res>0) {
        res = 0; // FIXME
    }
    CHECK(res,res);
    int k;
    for (k=0; k<mn; k++) {
        ipivp[k] = auxipiv[k];
    }
    free(auxipiv);
    OK
}


//////////////////// LU substitution /////////////////////////

int sgetrs_(char *trans, integer *n, integer *nrhs,
	real *a, integer *lda, integer *ipiv, real *b, integer *
	ldb, integer *info);

int luS_l_R(KOFMAT(a), KFVEC(ipiv), OFMAT(b)) {
  integer m = ar;
  integer n = ac;
  integer lda = aXc;
  integer mrhs = br;
  integer nrhs = bc;

  REQUIRES(m==n && m==mrhs && m==ipivn,BAD_SIZE);
  integer* auxipiv = (integer*)malloc(n*sizeof(integer));
  int k;
  for (k=0; k<n; k++) {
    auxipiv[k] = (integer)ipivp[k];
  }
  integer res;
  sgetrs_ ("N",&n,&nrhs,(/*no const (!?)*/ float*)ap,&lda,auxipiv,bp,&mrhs,&res);
  CHECK(res,res);
  free(auxipiv);
  OK
}


int cgetrs_(char *trans, integer *n, integer *nrhs,
	complex *a, integer *lda, integer *ipiv, complex *b,
	integer *ldb, integer *info);

int luS_l_C(KOQMAT(a), KFVEC(ipiv), OQMAT(b)) {
    integer m = ar;
    integer n = ac;
    integer lda = aXc;
    integer mrhs = br;
    integer nrhs = bc;

    REQUIRES(m==n && m==mrhs && m==ipivn,BAD_SIZE);
    integer* auxipiv = (integer*)malloc(n*sizeof(integer));
    int k;
    for (k=0; k<n; k++) {
        auxipiv[k] = (integer)ipivp[k];
    }
    integer res;
    cgetrs_ ("N",&n,&nrhs,(complex*)ap,&lda,auxipiv,bp,&mrhs,&res);
    CHECK(res,res);
    free(auxipiv);
    OK
}


//////////////////// LDL factorization /////////////////////////

int ssytrf_(char *uplo, integer *n, real *a, integer *lda, integer *ipiv,
            real *work, integer *lwork, integer *info);

int ldl_R(FVEC(ipiv), OFMAT(r)) {
    integer n = rr;
    REQUIRES(n>=1 && rc==n && ipivn == n, BAD_SIZE);
    DEBUGMSG("ldl_R");
    integer* auxipiv = (integer*)malloc(n*sizeof(integer));
    integer res;
    integer lda = rXc;
    integer lwork = -1;
    real ans;
    ssytrf_ ("L",&n,rp,&lda,auxipiv,&ans,&lwork,&res);
    lwork = ceil(ans);
    real* work = (real*)malloc(lwork*sizeof(real));
    ssytrf_ ("L",&n,rp,&lda,auxipiv,work,&lwork,&res);
    CHECK(res,res);
    int k;
    for (k=0; k<n; k++) {
        ipivp[k] = auxipiv[k];
    }
    free(auxipiv);
    free(work);
    OK
}


int chetrf_(char *uplo, integer *n, complex *a, integer *lda, integer *ipiv,
            complex *work, integer *lwork, integer *info);

int ldl_C(FVEC(ipiv), OQMAT(r)) {
    integer n = rr;
    REQUIRES(n>=1 && rc==n && ipivn == n, BAD_SIZE);
    DEBUGMSG("ldl_R");
    integer* auxipiv = (integer*)malloc(n*sizeof(integer));
    integer res;
    integer lda = rXc;
    integer lwork = -1;
    complex ans;
    chetrf_ ("L",&n,rp,&lda,auxipiv,&ans,&lwork,&res);
    lwork = ceil(ans.r);
    complex* work = (complex*)malloc(lwork*sizeof(complex));
    chetrf_ ("L",&n,rp,&lda,auxipiv,work,&lwork,&res);
    CHECK(res,res);
    int k;
    for (k=0; k<n; k++) {
        ipivp[k] = auxipiv[k];
    }
    free(auxipiv);
    free(work);
    OK

}

//////////////////// LDL solve /////////////////////////

int ssytrs_(char *uplo, integer *n, integer *nrhs, real *a, integer *lda,
            integer *ipiv, real *b, integer *ldb, integer *info);

int ldl_S_R(KOFMAT(a), KFVEC(ipiv), OFMAT(b)) {
  integer m = ar;
  integer n = ac;
  integer lda = aXc;
  integer mrhs = br;
  integer nrhs = bc;

  REQUIRES(m==n && m==mrhs && m==ipivn,BAD_SIZE);
  integer* auxipiv = (integer*)malloc(n*sizeof(integer));
  int k;
  for (k=0; k<n; k++) {
    auxipiv[k] = (integer)ipivp[k];
  }
  integer res;
  ssytrs_ ("L",&n,&nrhs,(/*no const (!?)*/ float*)ap,&lda,auxipiv,bp,&mrhs,&res);
  CHECK(res,res);
  free(auxipiv);
  OK
}


int chetrs_(char *uplo, integer *n, integer *nrhs, complex *a, integer *lda,
            integer *ipiv, complex *b, integer *ldb, integer *info);

int ldl_S_C(KOQMAT(a), KFVEC(ipiv), OQMAT(b)) {
    integer m = ar;
    integer n = ac;
    integer lda = aXc;
    integer mrhs = br;
    integer nrhs = bc;

    REQUIRES(m==n && m==mrhs && m==ipivn,BAD_SIZE);
    integer* auxipiv = (integer*)malloc(n*sizeof(integer));
    int k;
    for (k=0; k<n; k++) {
        auxipiv[k] = (integer)ipivp[k];
    }
    integer res;
    chetrs_ ("L",&n,&nrhs,(complex*)ap,&lda,auxipiv,bp,&mrhs,&res);
    CHECK(res,res);
    free(auxipiv);
    OK
}


//////////////////// Matrix Product /////////////////////////

void sgemm_(char *, char *, integer *, integer *, integer *,
            float *, const float *, integer *, const float *,
           integer *, float *, float *, integer *);

int multiplyF(int ta, int tb, KOFMAT(a),KOFMAT(b),OFMAT(r)) {
    DEBUGMSG("sgemm_");
    integer m = ta?ac:ar;
    integer n = tb?br:bc;
    integer k = ta?ar:ac;
    integer lda = aXc;
    integer ldb = bXc;
    integer ldc = rXc;
    float alpha = 1;
    float beta = 0;
    sgemm_(ta?"T":"N",tb?"T":"N",&m,&n,&k,&alpha,ap,&lda,bp,&ldb,&beta,rp,&ldc);
    OK
}

void cgemm_(char *, char *, integer *, integer *, integer *,
           complex *, const complex *, integer *, const complex *,
           integer *, complex *, complex *, integer *);

int multiplyQ(int ta, int tb, KOQMAT(a),KOQMAT(b),OQMAT(r)) {
    DEBUGMSG("cgemm_");
    integer m = ta?ac:ar;
    integer n = tb?br:bc;
    integer k = ta?ar:ac;
    integer lda = aXc;
    integer ldb = bXc;
    integer ldc = rXc;
    complex alpha = {1,0};
    complex beta = {0,0};
    cgemm_(ta?"T":"N",tb?"T":"N",&m,&n,&k,&alpha,
           ap,&lda,
           bp,&ldb,&beta,
           rp,&ldc);
    OK
}


#define MULT_IMP_VER(OP)        \
    { TRAV(r,i,j) {             \
        int k;                  \
        AT(r,i,j) = 0;          \
        for (k=0;k<ac;k++) {    \
            OP                  \
        }                       \
      }                         \
    }

#define MULT_IMP(M) {                                                                \
    if (m==1) {                                                                      \
        MULT_IMP_VER( AT(r,i,j) += AT(a,i,k) * AT(b,k,j); )                          \
    } else {                                                                         \
        MULT_IMP_VER( AT(r,i,j) = M(AT(r,i,j) + M(AT(a,i,k) * AT(b,k,j), m) , m) ; ) \
    } OK }

int multiplyI(int     m, KOIMAT(a), KOIMAT(b), OIMAT(r)) MULT_IMP(mod)
int multiplyL(int64_t m, KOLMAT(a), KOLMAT(b), OLMAT(r)) MULT_IMP(mod_l)

/////////////////////////////// inplace row ops ////////////////////////////////

#define AXPY_IMP {                    \
    int j;                            \
    for(j=j1; j<=j2; j++) {           \
        AT(r,i2,j) += a*AT(r,i1,j);   \
    } OK }

#define AXPY_MOD_IMP(M) {                                      \
    int j;                                                     \
    for(j=j1; j<=j2; j++) {                                    \
        AT(r,i2,j) = M(AT(r,i2,j) + M(a*AT(r,i1,j), m) , m);   \
    } OK }


#define SCAL_IMP {                    \
    int i,j;                          \
    for(i=i1; i<=i2; i++) {           \
        for(j=j1; j<=j2; j++) {       \
            AT(r,i,j) = a*AT(r,i,j);  \
            }                         \
    } OK }

#define SCAL_MOD_IMP(M) {                   \
    int i,j;                                \
    for(i=i1; i<=i2; i++) {                 \
        for(j=j1; j<=j2; j++) {             \
            AT(r,i,j) = M(a*AT(r,i,j) , m); \
            }                               \
    } OK }


#define SWAP_IMP(T)   {               \
    T aux;                            \
    int k;                            \
    if (i1 != i2) {                   \
        for (k=j1; k<=j2; k++) {      \
            aux = AT(r,i1,k);         \
            AT(r,i1,k) = AT(r,i2,k);  \
            AT(r,i2,k) = aux;         \
        }                             \
    } OK }


#define ROWOP_IMP(T) {                \
    T a = *pa;                        \
    switch(code) {                    \
        case 0:  AXPY_IMP             \
        case 1:  SCAL_IMP             \
        case 2:  SWAP_IMP(T)          \
        default: ERROR(BAD_CODE);     \
    }                                 \
}

#define ROWOP_MOD_IMP(T,M) {          \
    T a = *pa;                        \
    switch(code) {                    \
        case 0:  AXPY_MOD_IMP(M)      \
        case 1:  SCAL_MOD_IMP(M)      \
        case 2:  SWAP_IMP(T)          \
        default: ERROR(BAD_CODE);     \
    }                                 \
}


#define ROWOP(T) int rowop_##T(int code, T* pa, int i1, int i2, int j1, int j2, MATG(T,r)) ROWOP_IMP(T)

#define ROWOP_MOD(T,M) int rowop_mod_##T(T m, int code, T* pa, int i1, int i2, int j1, int j2, MATG(T,r)) ROWOP_MOD_IMP(T,M)

ROWOP(float)
ROWOP(TCS)
ROWOP(int32_t)
ROWOP(int64_t)
ROWOP_MOD(int32_t,mod)
ROWOP_MOD(int64_t,mod_l)

/////////////////////////////// inplace GEMM ////////////////////////////////

#define GEMM(T) int gemm_##T(VECG(T,c),MATG(T,a),MATG(T,b),MATG(T,r)) {  \
    T a = cp[0], b = cp[1];             \
    T t;                                \
    int k;                              \
    { TRAV(r,i,j) {                     \
      t = 0;                            \
      for(k=0; k<ac; k++) {             \
          t += AT(a,i,k) * AT(b,k,j);   \
      }                                 \
      AT(r,i,j) = b*AT(r,i,j) + a*t;    \
      }                                 \
    } OK }


GEMM(float)
/* GEMM(float) */
/* GEMM(TCD) */
GEMM(TCS)
GEMM(int32_t)
GEMM(int64_t)

#define GEMM_MOD(T,M) int gemm_mod_##T(T m, VECG(T,c),MATG(T,a),MATG(T,b),MATG(T,r)) {  \
    T a = cp[0], b = cp[1];                     \
    int k;                                      \
    T t;                                        \
    { TRAV(r,i,j) {                             \
      t = 0;                                    \
      for(k=0; k<ac; k++) {                     \
          t = M(t+M(AT(a,i,k) * AT(b,k,j)));    \
      }                                         \
      AT(r,i,j) = M(M(b*AT(r,i,j)) + M(a*t));   \
      }                                         \
    } OK }


#define MOD32(X) mod(X,m)
#define MOD64(X) mod_l(X,m)

GEMM_MOD(int32_t,MOD32)
GEMM_MOD(int64_t,MOD64)

////////////////// sparse matrix-product ///////////////////////////////////////


int smXv(KFVEC(vals),KIVEC(cols),KIVEC(rows),KFVEC(x),FVEC(r)) {
    int r, c;
    for (r = 0; r < rowsn - 1; r++) {
        rp[r] = 0;
        for (c = rowsp[r]; c < rowsp[r+1]; c++) {
            rp[r] += valsp[c-1] * xp[colsp[c-1]-1];
        }
    }
    OK
}

int smTXv(KFVEC(vals),KIVEC(cols),KIVEC(rows),KFVEC(x),FVEC(r)) {
    int r,c;
    for (c = 0; c < rn; c++) {
        rp[c] = 0;
    }
    for (r = 0; r < rowsn - 1; r++) {
        for (c = rowsp[r]; c < rowsp[r+1]; c++) {
            rp[colsp[c-1]-1] += valsp[c-1] * xp[r];
        }
    }
    OK
}


//////////////////////// extract /////////////////////////////////

#define EXTRACT_IMP {                      \
    int i,j,si,sj,ni,nj;                   \
    ni = modei ? in : ip[1]-ip[0]+1;       \
    nj = modej ? jn : jp[1]-jp[0]+1;       \
                                           \
    for (i=0; i<ni; i++) {                 \
        si = modei ? ip[i] : i+ip[0];      \
                                           \
        for (j=0; j<nj; j++) {             \
            sj = modej ? jp[j] : j+jp[0];  \
                                           \
            AT(r,i,j) = AT(m,si,sj);       \
        }                                  \
    } OK }

#define EXTRACT(T) int extract##T(int modei, int modej, KIVEC(i), KIVEC(j), KO##T##MAT(m), O##T##MAT(r)) EXTRACT_IMP

EXTRACT(F)
EXTRACT(Q)
EXTRACT(I)
EXTRACT(L)

//////////////////////// setRect /////////////////////////////////

#define SETRECT(T)                                            \
int setRect##T(int i, int j, KO##T##MAT(m), O##T##MAT(r)) {   \
    { TRAV(m,a,b) {                                           \
        int x = a+i, y = b+j;                                 \
        if(x>=0 && x<rr && y>=0 && y<rc) {                    \
            AT(r,x,y) = AT(m,a,b);                            \
        }                                                     \
      }                                                       \
    } OK }

SETRECT(F)
SETRECT(Q)
SETRECT(I)
SETRECT(L)

//////////////////////// remap /////////////////////////////////

#define REMAP_IMP                                               \
    REQUIRES(ir==jr && ic==jc && ir==rr && ic==rc ,BAD_SIZE);   \
    { TRAV(r,a,b) { AT(r,a,b) = AT(m,AT(i,a,b),AT(j,a,b)); }    \
    }                                                           \
    OK

/* int remapD(KOIMAT(i), KOIMAT(j), KOFMAT(m), OFMAT(r)) { */
/*     REMAP_IMP */
/* } */

int remapF(KOIMAT(i), KOIMAT(j), KOFMAT(m), OFMAT(r)) {
    REMAP_IMP
}

int remapI(KOIMAT(i), KOIMAT(j), KOIMAT(m), OIMAT(r)) {
    REMAP_IMP
}

int remapL(KOIMAT(i), KOIMAT(j), KOLMAT(m), OLMAT(r)) {
    REMAP_IMP
}

/* int remapC(KOIMAT(i), KOIMAT(j), KOQMAT(m), OQMAT(r)) { */
/*     REMAP_IMP */
/* } */

int remapQ(KOIMAT(i), KOIMAT(j), KOQMAT(m), OQMAT(r)) {
    REMAP_IMP
}

////////////////////////////////////////////////////////////////////////////////

int saveMatrix(char * file, char * format, KOFMAT(a)){
    FILE * fp;
    fp = fopen (file, "w");
    int r, c;
    for (r=0;r<ar; r++) {
        for (c=0; c<ac; c++) {
            fprintf(fp,format,AT(a,r,c));
            if (c<ac-1) {
                fprintf(fp," ");
            } else {
                fprintf(fp,"\n");
            }
        }
    }
    fclose(fp);
    OK
}

