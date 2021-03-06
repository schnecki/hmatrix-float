{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns  #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

-----------------------------------------------------------------------------
-- |
-- Module      :  Numeric.LinearAlgebra.LAPACK
-- Copyright   :  (c) Alberto Ruiz 2006-14
-- License     :  BSD3
-- Maintainer  :  Alberto Ruiz
-- Stability   :  provisional
--
-- Functional interface to selected LAPACK functions (<http://www.netlib.org/lapack>).
--
-----------------------------------------------------------------------------


module Internal.LAPACK where

import           Data.Bifunctor      (first)

import           Control.Monad       (when)
import           Foreign.C.Types
import           Foreign.Ptr         (nullPtr)
import           Internal.Conversion
import           Internal.Devel
import           Internal.Element
import           Internal.Matrix     hiding (( # ), ( #! ))
import           Internal.Vector
import           System.IO.Unsafe    (unsafePerformIO)

-----------------------------------------------------------------------------------

infixr 1 #
a # b = apply a b
{-# INLINE (#) #-}

a #! b = a # b # id
{-# INLINE (#!) #-}

-----------------------------------------------------------------------------------

type TMMM t = t ::> t ::> t ::> Ok

type F = Float
type Q = Complex Float

-- foreign import ccall unsafe "multiplyR" dgemmc :: CInt -> CInt -> TMMM R
-- foreign import ccall unsafe "multiplyC" zgemmc :: CInt -> CInt -> TMMM C
foreign import ccall unsafe "multiplyF" sgemmc :: CInt -> CInt -> TMMM F
foreign import ccall unsafe "multiplyQ" cgemmc :: CInt -> CInt -> TMMM Q
foreign import ccall unsafe "multiplyI" c_multiplyI :: I -> TMMM I
foreign import ccall unsafe "multiplyL" c_multiplyL :: Z -> TMMM Z

isT (rowOrder -> False) = 0
isT _                   = 1

tt x@(rowOrder -> False) = x
tt x                     = trans x

multiplyAux f st a b = unsafePerformIO $ do
    when (cols a /= rows b) $ error $ "inconsistent dimensions in matrix product "++
                                       show (rows a,cols a) ++ " x " ++ show (rows b, cols b)
    s <- createMatrix ColumnMajor (rows a) (cols b)
    ((tt a) # (tt b) #! s) (f (isT a) (isT b)) #| st
    return s

-- -- | Matrix product based on BLAS's /dgemm/.
-- multiplyR :: Matrix Float -> Matrix Float -> Matrix Float
-- multiplyR = multiplyF

-- -- | Matrix product based on BLAS's /zgemm/.
-- multiplyC :: Matrix (Complex Float) -> Matrix (Complex Float) -> Matrix (Complex Float)
-- multiplyC = multiplyQ

-- | Matrix product based on BLAS's /sgemm/.
multiplyF :: Matrix Float -> Matrix Float -> Matrix Float
multiplyF a b = multiplyAux sgemmc "sgemmc" a b

-- | Matrix product based on BLAS's /cgemm/.
multiplyQ :: Matrix (Complex Float) -> Matrix (Complex Float) -> Matrix (Complex Float)
multiplyQ a b = multiplyAux cgemmc "cgemmc" a b

multiplyI :: I -> Matrix CInt -> Matrix CInt -> Matrix CInt
multiplyI m a b = unsafePerformIO $ do
    when (cols a /= rows b) $ error $
        "inconsistent dimensions in matrix product "++ shSize a ++ " x " ++ shSize b
    s <- createMatrix ColumnMajor (rows a) (cols b)
    (a # b #! s) (c_multiplyI m) #|"c_multiplyI"
    return s

multiplyL :: Z -> Matrix Z -> Matrix Z -> Matrix Z
multiplyL m a b = unsafePerformIO $ do
    when (cols a /= rows b) $ error $
        "inconsistent dimensions in matrix product "++ shSize a ++ " x " ++ shSize b
    s <- createMatrix ColumnMajor (rows a) (cols b)
    (a # b #! s) (c_multiplyL m) #|"c_multiplyL"
    return s

-----------------------------------------------------------------------------

type TSVD t = t ::> t ::> R :> t ::> Ok

-- foreign import ccall unsafe "svd_l_R" dgesvd :: TSVD R
foreign import ccall unsafe "svd_l_S" sgesvd :: TSVD F
-- foreign import ccall unsafe "svd_l_C" zgesvd :: TSVD C
foreign import ccall unsafe "svd_l_C" cgesvd :: TSVD C
-- foreign import ccall unsafe "svd_l_Rdd" dgesdd :: TSVD R
foreign import ccall unsafe "svd_l_Rdd" sgesdd :: TSVD R
-- foreign import ccall unsafe "svd_l_Cdd" zgesdd :: TSVD C
foreign import ccall unsafe "svd_l_Cdd" cgesdd :: TSVD C

-- | Full SVD of a real matrix using LAPACK's /dgesvd/.
svdR :: Matrix Float -> (Matrix Float, Vector Float, Matrix Float)
svdR = svdAux sgesvd "svdS"

-- | Full SVD of a real matrix using LAPACK's /dgesdd/.
svdRd :: Matrix Float -> (Matrix Float, Vector Float, Matrix Float)
svdRd = svdAux sgesdd "svdSdd"

-- | Full SVD of a complex matrix using LAPACK's /zgesvd/.
svdC :: Matrix (Complex Float) -> (Matrix (Complex Float), Vector Float, Matrix (Complex Float))
svdC = svdAux cgesvd "svdC"

-- | Full SVD of a complex matrix using LAPACK's /zgesdd/.
svdCd :: Matrix (Complex Float) -> (Matrix (Complex Float), Vector Float, Matrix (Complex Float))
svdCd = svdAux cgesdd "svdCdd"

svdAux f st x = unsafePerformIO $ do
    a <- copy ColumnMajor x
    u <- createMatrix ColumnMajor r r
    s <- createVector (min r c)
    v <- createMatrix ColumnMajor c c
    (a # u # s #! v) f #| st
    return (u,s,v)
  where
    r = rows x
    c = cols x


-- | Thin SVD of a real matrix, using LAPACK's /dgesvd/ with jobu == jobvt == \'S\'.
thinSVDR :: Matrix Float -> (Matrix Float, Vector Float, Matrix Float)
thinSVDR = thinSVDAux sgesvd "thinSVDR"

-- | Thin SVD of a complex matrix, using LAPACK's /zgesvd/ with jobu == jobvt == \'S\'.
thinSVDC :: Matrix (Complex Float) -> (Matrix (Complex Float), Vector Float, Matrix (Complex Float))
thinSVDC = thinSVDAux cgesvd "thinSVDC"

-- | Thin SVD of a real matrix, using LAPACK's /dgesdd/ with jobz == \'S\'.
thinSVDRd :: Matrix Float -> (Matrix Float, Vector Float, Matrix Float)
thinSVDRd = thinSVDAux sgesdd "thinSVDRdd"

-- | Thin SVD of a complex matrix, using LAPACK's /zgesdd/ with jobz == \'S\'.
thinSVDCd :: Matrix (Complex Float) -> (Matrix (Complex Float), Vector Float, Matrix (Complex Float))
thinSVDCd = thinSVDAux cgesdd "thinSVDCdd"

thinSVDAux f st x = unsafePerformIO $ do
    a <- copy ColumnMajor x
    u <- createMatrix ColumnMajor r q
    s <- createVector q
    v <- createMatrix ColumnMajor q c
    (a # u # s #! v) f #| st
    return (u,s,v)
  where
    r = rows x
    c = cols x
    q = min r c


-- | Singular values of a real matrix, using LAPACK's /dgesvd/ with jobu == jobvt == \'N\'.
svR :: Matrix Float -> Vector Float
svR = svAux sgesvd "svR"

-- | Singular values of a complex matrix, using LAPACK's /zgesvd/ with jobu == jobvt == \'N\'.
svC :: Matrix (Complex Float) -> Vector Float
svC = svAux cgesvd "svC"

-- | Singular values of a real matrix, using LAPACK's /dgesdd/ with jobz == \'N\'.
svRd :: Matrix Float -> Vector Float
svRd = svAux sgesdd "svRd"

-- | Singular values of a complex matrix, using LAPACK's /zgesdd/ with jobz == \'N\'.
svCd :: Matrix (Complex Float) -> Vector Float
svCd = svAux cgesdd "svCd"

svAux f st x = unsafePerformIO $ do
    a <- copy ColumnMajor x
    s <- createVector q
    (a #! s) g #| st
    return s
  where
    r = rows x
    c = cols x
    q = min r c
    g ra ca xra xca pa nb pb = f ra ca xra xca pa 0 0 0 0 nullPtr nb pb 0 0 0 0 nullPtr


-- | Singular values and all right singular vectors of a real matrix, using LAPACK's /dgesvd/ with jobu == \'N\' and jobvt == \'A\'.
rightSVR :: Matrix Float -> (Vector Float, Matrix Float)
rightSVR = rightSVAux sgesvd "rightSVR"

-- | Singular values and all right singular vectors of a complex matrix, using LAPACK's /zgesvd/ with jobu == \'N\' and jobvt == \'A\'.
rightSVC :: Matrix (Complex Float) -> (Vector Float, Matrix (Complex Float))
rightSVC = rightSVAux cgesvd "rightSVC"

rightSVAux f st x = unsafePerformIO $ do
    a <- copy ColumnMajor x
    s <- createVector q
    v <- createMatrix ColumnMajor c c
    (a # s #! v) g #| st
    return (s,v)
  where
    r = rows x
    c = cols x
    q = min r c
    g ra ca xra xca pa = f ra ca xra xca pa 0 0 0 0 nullPtr


-- | Singular values and all left singular vectors of a real matrix, using LAPACK's /dgesvd/  with jobu == \'A\' and jobvt == \'N\'.
leftSVR :: Matrix Float -> (Matrix Float, Vector Float)
leftSVR = leftSVAux sgesvd "leftSVR"

-- | Singular values and all left singular vectors of a complex matrix, using LAPACK's /zgesvd/ with jobu == \'A\' and jobvt == \'N\'.
leftSVC :: Matrix (Complex Float) -> (Matrix (Complex Float), Vector Float)
leftSVC = leftSVAux cgesvd "leftSVC"

leftSVAux f st x = unsafePerformIO $ do
    a <- copy ColumnMajor x
    u <- createMatrix ColumnMajor r r
    s <- createVector q
    (a # u #! s) g #| st
    return (u,s)
  where
    r = rows x
    c = cols x
    q = min r c
    g ra ca xra xca pa ru cu xru xcu pu nb pb = f ra ca xra xca pa ru cu xru xcu pu nb pb 0 0 0 0 nullPtr

-----------------------------------------------------------------------------

-- foreign import ccall unsafe "eig_l_R" dgeev :: R ::> R ::> C :> R ::> Ok
-- foreign import ccall unsafe "eig_l_G" dggev :: R ::> R ::> C :> R :> R ::> R ::> Ok
-- foreign import ccall unsafe "eig_l_C" zgeev :: C ::> C ::> C :> C ::> Ok
-- foreign import ccall unsafe "eig_l_GC" zggev :: C ::> C ::> C :> C :> C ::> C ::> Ok
-- foreign import ccall unsafe "eig_l_S" dsyev :: CInt -> R :> R ::> Ok
-- foreign import ccall unsafe "eig_l_H" zheev :: CInt -> R :> C ::> Ok

foreign import ccall unsafe "eig_l_R" sgeev :: R ::> R ::> C :> R ::> Ok
foreign import ccall unsafe "eig_l_G" sggev :: R ::> R ::> C :> R :> R ::> R ::> Ok
foreign import ccall unsafe "eig_l_C" cgeev :: C ::> C ::> C :> C ::> Ok
foreign import ccall unsafe "eig_l_GC" cggev :: C ::> C ::> C :> C :> C ::> C ::> Ok
foreign import ccall unsafe "eig_l_S" ssyev :: CInt -> R :> R ::> Ok
foreign import ccall unsafe "eig_l_H" cheev :: CInt -> R :> C ::> Ok


eigAux f st m = unsafePerformIO $ do
    a <- copy ColumnMajor m
    l <- createVector r
    v <- createMatrix ColumnMajor r r
    (a # l #! v) g #| st
    return (l,v)
  where
    r = rows m
    g ra ca xra xca pa = f ra ca xra xca pa 0 0 0 0 nullPtr


-- | Eigenvalues and right eigenvectors of a general complex matrix, using LAPACK's /zgeev/.
-- The eigenvectors are the columns of v. The eigenvalues are not sorted.
eigC :: Matrix (Complex Float) -> (Vector (Complex Float), Matrix (Complex Float))
eigC = eigAux cgeev "eigC"

eigOnlyAux f st m = unsafePerformIO $ do
    a <- copy ColumnMajor m
    l <- createVector r
    (a #! l) g #| st
    return l
  where
    r = rows m
    g ra ca xra xca pa nl pl = f ra ca xra xca pa 0 0 0 0 nullPtr nl pl 0 0 0 0 nullPtr

-- | Eigenvalues of a general complex matrix, using LAPACK's /zgeev/ with jobz == \'N\'.
-- The eigenvalues are not sorted.
eigOnlyC :: Matrix (Complex Float) -> Vector (Complex Float)
eigOnlyC = eigOnlyAux cgeev "eigOnlyC"

-- | Eigenvalues and right eigenvectors of a general real matrix, using LAPACK's /dgeev/.
-- The eigenvectors are the columns of v. The eigenvalues are not sorted.
eigR :: Matrix Float -> (Vector (Complex Float), Matrix (Complex Float))
eigR m = (s', v'')
    where (s,v) = eigRaux m
          s' = fixeig1 s
          v' = toRows $ trans v
          v'' = fromColumns $ fixeig (toList s') v'

eigRaux :: Matrix Float -> (Vector (Complex Float), Matrix Float)
eigRaux m = unsafePerformIO $ do
    a <- copy ColumnMajor m
    l <- createVector r
    v <- createMatrix ColumnMajor r r
    (a # l #! v) g #| "eigR"
    return (l,v)
  where
    r = rows m
    g ra ca xra xca pa = sgeev ra ca xra xca pa 0 0 0 0 nullPtr

fixeig1 s = toComplex' (subVector 0 r (asReal s), subVector r r (asReal s))
    where r = dim s

fixeig  []  _ =  []
fixeig [_] [v] = [comp' v]
fixeig ((r1:+i1):(r2:+i2):r) (v1:v2:vs)
    | r1 == r2 && i1 == (-i2) = toComplex' (v1,v2) : toComplex' (v1, mapVector negate v2) : fixeig r vs
    | otherwise = comp' v1 : fixeig ((r2:+i2):r) (v2:vs)
fixeig _ _ = error "fixeig with impossible inputs"

-- For dggev alpha(i) / beta(i), alpha(i+1) / beta(i+1) form a complex conjugate pair when Im alpha(i) != 0.
-- However, this does not lead to Re alpha(i) == Re alpha(i+1), since beta(i) and beta(i+1)
-- can be different. Therefore old 'fixeig' would fail for 'eigG'.
fixeigG  []  _  = []
fixeigG [_] [v] = [comp' v]
fixeigG ((_:+ai1) : an : as) (v1:v2:vs)
    | abs ai1 > 1e-13 = toComplex' (v1, v2) : toComplex' (v1, mapVector negate v2) : fixeigG as vs
    | otherwise = comp' v1 : fixeigG (an:as) (v2:vs)
fixeigG _ _ = error "fixeigG with impossible inputs"

-- | Eigenvalues of a general real matrix, using LAPACK's /dgeev/ with jobz == \'N\'.
-- The eigenvalues are not sorted.
eigOnlyR :: Matrix Float -> Vector (Complex Float)
eigOnlyR = fixeig1 . eigOnlyAux sgeev "eigOnlyR"

-- | Generalized eigenvalues and right eigenvectors of a pair of real matrices, using LAPACK's /dggev/.
-- The eigenvectors are the columns of v. The eigenvalues are represented as alphas / betas and not sorted.
eigG :: Matrix Float -> Matrix Float -> (Vector (Complex Float), Vector Float, Matrix (Complex Float))
eigG a b = (alpha', beta, v'')
  where
    (alpha, beta, v) = eigGaux sggev a b "eigG"
    alpha' = fixeig1 alpha
    v' = toRows $ trans v
    v'' = fromColumns $ fixeigG (toList alpha') v'

eigGaux f ma mb st = unsafePerformIO $ do
    a <- copy ColumnMajor ma
    b <- copy ColumnMajor mb
    alpha <- createVector r
    beta <- createVector r
    vr <- createMatrix ColumnMajor r r

    (a # b # alpha # beta #! vr) g #| st

    return (alpha, beta, vr)
  where
    r = rows ma
    g ar ac xra xca pa br bc xrb xcb pb alphan palpha betan pbeta = f ar ac xra xca pa br bc xrb xcb pb alphan palpha betan pbeta 0 0 0 0 nullPtr

eigGOnlyAux f ma mb st = unsafePerformIO $ do
    a <- copy ColumnMajor ma
    b <- copy ColumnMajor mb
    alpha <- createVector r
    beta <- createVector r

    (a # b # alpha #! beta) g #| st

    return (alpha, beta)
  where
    r = rows ma
    g ar ac xra xca pa br bc xrb xcb pb alphan palpha betan pbeta = f ar ac xra xca pa br bc xrb xcb pb alphan palpha betan pbeta 0 0 0 0 nullPtr 0 0 0 0 nullPtr

-- | Generalized eigenvalues and right eigenvectors of a pair of complex matrices, using LAPACK's /zggev/.
-- The eigenvectors are the columns of v. The eigenvalues are represented as alphas / betas and not sorted.
eigGC :: Matrix (Complex Float) -> Matrix (Complex Float) -> (Vector (Complex Float), Vector (Complex Float), Matrix (Complex Float))
eigGC a b = eigGaux cggev a b "eigGC"

eigOnlyG :: Matrix Float -> Matrix Float -> (Vector (Complex Float), Vector Float)
eigOnlyG a b = first fixeig1 $ eigGOnlyAux sggev a b "eigOnlyG"

eigOnlyGC :: Matrix (Complex Float) -> Matrix (Complex Float) -> (Vector (Complex Float), Vector (Complex Float))
eigOnlyGC a b = eigGOnlyAux cggev a b "eigOnlyGC"

-----------------------------------------------------------------------------

eigSHAux f st m = unsafePerformIO $ do
    l <- createVector r
    v <- copy ColumnMajor m
    (l #! v) f #| st
    return (l,v)
  where
    r = rows m

-- | Eigenvalues and right eigenvectors of a symmetric real matrix, using LAPACK's /dsyev/.
-- The eigenvectors are the columns of v.
-- The eigenvalues are sorted in descending order (use 'eigS'' for ascending order).
eigS :: Matrix Float -> (Vector Float, Matrix Float)
eigS m = (s', fliprl v)
    where (s,v) = eigS' m
          s' = fromList . reverse . toList $  s

-- | 'eigS' in ascending order
eigS' :: Matrix Float -> (Vector Float, Matrix Float)
eigS' = eigSHAux (ssyev 1) "eigS'"

-- | Eigenvalues and right eigenvectors of a hermitian complex matrix, using LAPACK's /zheev/.
-- The eigenvectors are the columns of v.
-- The eigenvalues are sorted in descending order (use 'eigH'' for ascending order).
eigH :: Matrix (Complex Float) -> (Vector Float, Matrix (Complex Float))
eigH m = (s', fliprl v)
  where
    (s,v) = eigH' m
    s' = fromList . reverse . toList $  s

-- | 'eigH' in ascending order
eigH' :: Matrix (Complex Float) -> (Vector Float, Matrix (Complex Float))
eigH' = eigSHAux (cheev 1) "eigH'"


-- | Eigenvalues of a symmetric real matrix, using LAPACK's /dsyev/ with jobz == \'N\'.
-- The eigenvalues are sorted in descending order.
eigOnlyS :: Matrix Float -> Vector Float
eigOnlyS = vrev . fst. eigSHAux (ssyev 0) "eigS'"

-- | Eigenvalues of a hermitian complex matrix, using LAPACK's /zheev/ with jobz == \'N\'.
-- The eigenvalues are sorted in descending order.
eigOnlyH :: Matrix (Complex Float) -> Vector Float
eigOnlyH = vrev . fst. eigSHAux (cheev 0) "eigH'"

vrev = flatten . flipud . reshape 1

-----------------------------------------------------------------------------
-- foreign import ccall unsafe "linearSolveR_l" dgesv :: R ::> R ::> Ok
-- foreign import ccall unsafe "linearSolveC_l" zgesv :: C ::> C ::> Ok
foreign import ccall unsafe "linearSolveR_l" sgesv :: R ::> R ::> Ok
foreign import ccall unsafe "linearSolveC_l" cgesv :: C ::> C ::> Ok

linearSolveSQAux g f st a b
    | n1==n2 && n1==r = unsafePerformIO . g $ do
        a' <- copy ColumnMajor a
        s  <- copy ColumnMajor b
        (a' #! s) f #| st
        return s
    | otherwise = error $ st ++ " of nonsquare matrix"
  where
    n1 = rows a
    n2 = cols a
    r  = rows b

-- | Solve a real linear system (for square coefficient matrix and several right-hand sides) using the LU decomposition, based on LAPACK's /dgesv/. For underconstrained or overconstrained systems use 'linearSolveLSR' or 'linearSolveSVDR'. See also 'lusR'.
linearSolveR :: Matrix Float -> Matrix Float -> Matrix Float
linearSolveR a b = linearSolveSQAux id sgesv "linearSolveR" a b

mbLinearSolveR :: Matrix Float -> Matrix Float -> Maybe (Matrix Float)
mbLinearSolveR a b = linearSolveSQAux mbCatch sgesv "linearSolveR" a b


-- | Solve a complex linear system (for square coefficient matrix and several right-hand sides) using the LU decomposition, based on LAPACK's /zgesv/. For underconstrained or overconstrained systems use 'linearSolveLSC' or 'linearSolveSVDC'. See also 'lusC'.
linearSolveC :: Matrix (Complex Float) -> Matrix (Complex Float) -> Matrix (Complex Float)
linearSolveC a b = linearSolveSQAux id cgesv "linearSolveC" a b

mbLinearSolveC :: Matrix (Complex Float) -> Matrix (Complex Float) -> Maybe (Matrix (Complex Float))
mbLinearSolveC a b = linearSolveSQAux mbCatch cgesv "linearSolveC" a b

--------------------------------------------------------------------------------
foreign import ccall unsafe "cholSolveR_l" dpotrs  :: R ::> R ::> Ok
foreign import ccall unsafe "cholSolveC_l" zpotrs  :: C ::> C ::> Ok


linearSolveSQAux2 g f st a b
    | n1==n2 && n1==r = unsafePerformIO . g $ do
        s <- copy ColumnMajor b
        (a #! s) f #| st
        return s
    | otherwise = error $ st ++ " of nonsquare matrix"
  where
    n1 = rows a
    n2 = cols a
    r  = rows b

-- | Solves a symmetric positive definite system of linear equations using a precomputed Cholesky factorization obtained by 'cholS'.
cholSolveR :: Matrix Float -> Matrix Float -> Matrix Float
cholSolveR a b = linearSolveSQAux2 id dpotrs "cholSolveR" (fmat a) b

-- | Solves a Hermitian positive definite system of linear equations using a precomputed Cholesky factorization obtained by 'cholH'.
cholSolveC :: Matrix (Complex Float) -> Matrix (Complex Float) -> Matrix (Complex Float)
cholSolveC a b = linearSolveSQAux2 id zpotrs "cholSolveC" (fmat a) b

--------------------------------------------------------------------------------
foreign import ccall unsafe "triSolveR_l_u" dtrtrs_u  :: R ::> R ::> Ok
foreign import ccall unsafe "triSolveC_l_u" ztrtrs_u  :: C ::> C ::> Ok
foreign import ccall unsafe "triSolveR_l_l" dtrtrs_l  :: R ::> R ::> Ok
foreign import ccall unsafe "triSolveC_l_l" ztrtrs_l  :: C ::> C ::> Ok


linearSolveTRAux2 g f st a b
    | n1==n2 && n1==r = unsafePerformIO . g $ do
        s <- copy ColumnMajor b
        (a #! s) f #| st
        return s
    | otherwise = error $ st ++ " of nonsquare matrix"
  where
    n1 = rows a
    n2 = cols a
    r  = rows b

data UpLo = Lower | Upper

-- | Solves a triangular system of linear equations.
triSolveR :: UpLo -> Matrix Float -> Matrix Float -> Matrix Float
triSolveR Lower a b = linearSolveTRAux2 id dtrtrs_l "triSolveR" (fmat a) b
triSolveR Upper a b = linearSolveTRAux2 id dtrtrs_u "triSolveR" (fmat a) b

-- | Solves a triangular system of linear equations.
triSolveC :: UpLo -> Matrix (Complex Float) -> Matrix (Complex Float) -> Matrix (Complex Float)
triSolveC Lower a b = linearSolveTRAux2 id ztrtrs_l "triSolveC" (fmat a) b
triSolveC Upper a b = linearSolveTRAux2 id ztrtrs_u "triSolveC" (fmat a) b

--------------------------------------------------------------------------------
foreign import ccall unsafe "triDiagSolveR_l" dgttrs  :: R :> R :> R :> R ::> Ok
foreign import ccall unsafe "triDiagSolveC_l" zgttrs  :: C :> C :> C :> C ::> Ok

linearSolveGTAux2 g f st dl d du b
    | ndl  == nd - 1 &&
      ndu  == nd - 1 &&
      nd   == r = unsafePerformIO . g $ do
        dl' <- head . toRows <$> copy ColumnMajor (fromRows [dl])
        du' <- head . toRows <$> copy ColumnMajor (fromRows [du])
        s <- copy ColumnMajor b
        (dl' # d # du' #! s) f #| st
        return s
    | otherwise = error $ st ++ " of nonsquare matrix"
  where
    ndl  = dim dl
    nd   = dim d
    ndu  = dim du
    r    = rows b

-- | Solves a tridiagonal system of linear equations.
triDiagSolveR dl d du b = linearSolveGTAux2 id dgttrs "triDiagSolveR" dl d du b
triDiagSolveC dl d du b = linearSolveGTAux2 id zgttrs "triDiagSolveC" dl d du b

-----------------------------------------------------------------------------------

-- foreign import ccall unsafe "linearSolveLSR_l"   dgels ::           R ::> R ::> Ok
-- foreign import ccall unsafe "linearSolveLSC_l"   zgels ::           C ::> C ::> Ok
-- foreign import ccall unsafe "linearSolveSVDR_l" dgelss :: Double -> R ::> R ::> Ok
-- foreign import ccall unsafe "linearSolveSVDC_l" zgelss :: Double -> C ::> C ::> Ok

foreign import ccall unsafe "linearSolveLSR_l"   sgels ::          R ::> R ::> Ok
foreign import ccall unsafe "linearSolveLSC_l"   cgels ::          C ::> C ::> Ok
foreign import ccall unsafe "linearSolveSVDR_l" sgelss :: Float -> R ::> R ::> Ok
foreign import ccall unsafe "linearSolveSVDC_l" cgelss :: Float -> C ::> C ::> Ok


linearSolveAux f st a b
    | m == rows b = unsafePerformIO $ do
        a' <- copy ColumnMajor a
        r  <- createMatrix ColumnMajor (max m n) nrhs
        setRect 0 0 b r
        (a' #! r) f #| st
        return r
    | otherwise = error $ "different number of rows in linearSolve ("++st++")"
  where
    m = rows a
    n = cols a
    nrhs = cols b

-- | Least squared error solution of an overconstrained real linear system, or the minimum norm solution of an underconstrained system, using LAPACK's /dgels/. For rank-deficient systems use 'linearSolveSVDR'.
linearSolveLSR :: Matrix Float -> Matrix Float -> Matrix Float
linearSolveLSR a b = subMatrix (0,0) (cols a, cols b) $
                     linearSolveAux sgels "linearSolverLSR" a b

-- | Least squared error solution of an overconstrained complex linear system, or the minimum norm solution of an underconstrained system, using LAPACK's /zgels/. For rank-deficient systems use 'linearSolveSVDC'.
linearSolveLSC :: Matrix (Complex Float) -> Matrix (Complex Float) -> Matrix (Complex Float)
linearSolveLSC a b = subMatrix (0,0) (cols a, cols b) $
                     linearSolveAux cgels "linearSolveLSC" a b

-- | Minimum norm solution of a general real linear least squares problem Ax=B using the SVD, based on LAPACK's /dgelss/. Admits rank-deficient systems but it is slower than 'linearSolveLSR'. The effective rank of A is determined by treating as zero those singular valures which are less than rcond times the largest singular value. If rcond == Nothing machine precision is used.
linearSolveSVDR :: Maybe Float   -- ^ rcond
                -> Matrix Float  -- ^ coefficient matrix
                -> Matrix Float  -- ^ right hand sides (as columns)
                -> Matrix Float  -- ^ solution vectors (as columns)
linearSolveSVDR (Just rcond) a b = subMatrix (0,0) (cols a, cols b) $
                                   linearSolveAux (sgelss rcond) "linearSolveSVDR" a b
linearSolveSVDR Nothing a b = linearSolveSVDR (Just (-1)) a b

-- | Minimum norm solution of a general complex linear least squares problem Ax=B using the SVD, based on LAPACK's /zgelss/. Admits rank-deficient systems but it is slower than 'linearSolveLSC'. The effective rank of A is determined by treating as zero those singular valures which are less than rcond times the largest singular value. If rcond == Nothing machine precision is used.
linearSolveSVDC :: Maybe Float            -- ^ rcond
                -> Matrix (Complex Float) -- ^ coefficient matrix
                -> Matrix (Complex Float) -- ^ right hand sides (as columns)
                -> Matrix (Complex Float) -- ^ solution vectors (as columns)
linearSolveSVDC (Just rcond) a b = subMatrix (0,0) (cols a, cols b) $
                                   linearSolveAux (cgelss rcond) "linearSolveSVDC" a b
linearSolveSVDC Nothing a b = linearSolveSVDC (Just (-1)) a b

-----------------------------------------------------------------------------------

-- foreign import ccall unsafe "chol_l_H" zpotrf :: C ::> Ok
-- foreign import ccall unsafe "chol_l_S" dpotrf :: R ::> Ok

foreign import ccall unsafe "chol_l_H" cpotrf :: C ::> Ok
foreign import ccall unsafe "chol_l_S" spotrf :: R ::> Ok

cholAux f st a = do
    r <- copy ColumnMajor a
    (r # id) f #| st
    return r

-- | Cholesky factorization of a complex Hermitian positive definite matrix, using LAPACK's /zpotrf/.
cholH :: Matrix (Complex Float) -> Matrix (Complex Float)
cholH = unsafePerformIO . cholAux cpotrf "cholH"

-- | Cholesky factorization of a real symmetric positive definite matrix, using LAPACK's /dpotrf/.
cholS :: Matrix Float -> Matrix Float
cholS =  unsafePerformIO . cholAux spotrf "cholS"

-- | Cholesky factorization of a complex Hermitian positive definite matrix, using LAPACK's /zpotrf/ ('Maybe' version).
mbCholH :: Matrix (Complex Float) -> Maybe (Matrix (Complex Float))
mbCholH = unsafePerformIO . mbCatch . cholAux cpotrf "cholH"

-- | Cholesky factorization of a real symmetric positive definite matrix, using LAPACK's /dpotrf/  ('Maybe' version).
mbCholS :: Matrix Float -> Maybe (Matrix Float)
mbCholS =  unsafePerformIO . mbCatch . cholAux spotrf "cholS"

-----------------------------------------------------------------------------------

type TMVM t = t ::> t :> t ::> Ok

-- foreign import ccall unsafe "qr_l_R" dgeqr2 :: R :> R ::> Ok
-- foreign import ccall unsafe "qr_l_C" zgeqr2 :: C :> C ::> Ok

foreign import ccall unsafe "qr_l_R" sgeqr2 :: R :> R ::> Ok
foreign import ccall unsafe "qr_l_C" cgeqr2 :: C :> C ::> Ok


-- | QR factorization of a real matrix, using LAPACK's /dgeqr2/.
qrR :: Matrix Float -> (Matrix Float, Vector Float)
qrR = qrAux sgeqr2 "qrR"

-- | QR factorization of a complex matrix, using LAPACK's /zgeqr2/.
qrC :: Matrix (Complex Float) -> (Matrix (Complex Float), Vector (Complex Float))
qrC = qrAux cgeqr2 "qrC"

qrAux f st a = unsafePerformIO $ do
    r <- copy ColumnMajor a
    tau <- createVector mn
    (tau #! r) f #| st
    return (r,tau)
  where
    m = rows a
    n = cols a
    mn = min m n

-- foreign import ccall unsafe "c_dorgqr" dorgqr :: R :> R ::> Ok
-- foreign import ccall unsafe "c_zungqr" zungqr :: C :> C ::> Ok

foreign import ccall unsafe "c_dorgqr" sorgqr :: R :> R ::> Ok
foreign import ccall unsafe "c_zungqr" cungqr :: C :> C ::> Ok


-- | build rotation from reflectors
qrgrR :: Int -> (Matrix Float, Vector Float) -> Matrix Float
qrgrR = qrgrAux sorgqr "qrgrR"
-- | build rotation from reflectors
qrgrC :: Int -> (Matrix (Complex Float), Vector (Complex Float)) -> Matrix (Complex Float)
qrgrC = qrgrAux cungqr "qrgrC"

qrgrAux f st n (a, tau) = unsafePerformIO $ do
    res <- copy ColumnMajor (subMatrix (0,0) (rows a,n) a)
    ((subVector 0 n tau') #! res) f #| st
    return res
  where
    tau' = vjoin [tau, constantD 0 n]

-----------------------------------------------------------------------------------
-- foreign import ccall unsafe "hess_l_R" dgehrd :: R :> R ::> Ok
-- foreign import ccall unsafe "hess_l_C" zgehrd :: C :> C ::> Ok

foreign import ccall unsafe "hess_l_R" sgehrd :: R :> R ::> Ok
foreign import ccall unsafe "hess_l_C" cgehrd :: C :> C ::> Ok

-- | Hessenberg factorization of a square real matrix, using LAPACK's /dgehrd/.
hessR :: Matrix Float -> (Matrix Float, Vector Float)
hessR = hessAux sgehrd "hessR"

-- | Hessenberg factorization of a square complex matrix, using LAPACK's /zgehrd/.
hessC :: Matrix (Complex Float) -> (Matrix (Complex Float), Vector (Complex Float))
hessC = hessAux cgehrd "hessC"

hessAux f st a = unsafePerformIO $ do
    r <- copy ColumnMajor a
    tau <- createVector (mn-1)
    (tau #! r) f #| st
    return (r,tau)
  where
    m = rows a
    n = cols a
    mn = min m n

-----------------------------------------------------------------------------------
-- foreign import ccall unsafe "schur_l_R" dgees :: R ::> R ::> Ok
-- foreign import ccall unsafe "schur_l_C" zgees :: C ::> C ::> Ok

foreign import ccall unsafe "schur_l_R" sgees :: R ::> R ::> Ok
foreign import ccall unsafe "schur_l_C" cgees :: C ::> C ::> Ok


-- | Schur factorization of a square real matrix, using LAPACK's /dgees/.
schurR :: Matrix Float -> (Matrix Float, Matrix Float)
schurR = schurAux sgees "schurR"

-- | Schur factorization of a square complex matrix, using LAPACK's /zgees/.
schurC :: Matrix (Complex Float) -> (Matrix (Complex Float), Matrix (Complex Float))
schurC = schurAux cgees "schurC"

schurAux f st a = unsafePerformIO $ do
    u <- createMatrix ColumnMajor n n
    s <- copy ColumnMajor a
    (u #! s) f #| st
    return (u,s)
  where
    n = rows a

-----------------------------------------------------------------------------------
-- foreign import ccall unsafe "lu_l_R" dgetrf :: R :> R ::> Ok
-- foreign import ccall unsafe "lu_l_C" zgetrf :: R :> C ::> Ok

foreign import ccall unsafe "lu_l_R" sgetrf :: R :> R ::> Ok
foreign import ccall unsafe "lu_l_C" cgetrf :: R :> C ::> Ok

-- | LU factorization of a general real matrix, using LAPACK's /dgetrf/.
luR :: Matrix Float -> (Matrix Float, [Int])
luR = luAux sgetrf "luR"

-- | LU factorization of a general complex matrix, using LAPACK's /zgetrf/.
luC :: Matrix (Complex Float) -> (Matrix (Complex Float), [Int])
luC = luAux cgetrf "luC"

luAux f st a = unsafePerformIO $ do
    lu <- copy ColumnMajor a
    piv <- createVector (min n m)
    (piv #! lu) f #| st
    return (lu, map (pred.round) (toList piv))
  where
    n = rows a
    m = cols a

-----------------------------------------------------------------------------------

-- foreign import ccall unsafe "luS_l_R" dgetrs :: R ::> R :> R ::> Ok
-- foreign import ccall unsafe "luS_l_C" zgetrs :: C ::> R :> C ::> Ok

foreign import ccall unsafe "luS_l_R" sgetrs :: R ::> R :> R ::> Ok
foreign import ccall unsafe "luS_l_C" cgetrs :: C ::> R :> C ::> Ok


-- | Solve a real linear system from a precomputed LU decomposition ('luR'), using LAPACK's /dgetrs/.
lusR :: Matrix Float -> [Int] -> Matrix Float -> Matrix Float
lusR a piv b = lusAux sgetrs "lusR" (fmat a) piv b

-- | Solve a complex linear system from a precomputed LU decomposition ('luC'), using LAPACK's /zgetrs/.
lusC :: Matrix (Complex Float) -> [Int] -> Matrix (Complex Float) -> Matrix (Complex Float)
lusC a piv b = lusAux cgetrs "lusC" (fmat a) piv b

lusAux f st a piv b
    | n1==n2 && n2==n =unsafePerformIO $ do
         x <- copy ColumnMajor b
         (a # piv' #! x) f #| st
         return x
    | otherwise = error st
  where
    n1 = rows a
    n2 = cols a
    n = rows b
    piv' = fromList (map (fromIntegral.succ) piv) :: Vector Float

-----------------------------------------------------------------------------------
-- foreign import ccall unsafe "ldl_R" dsytrf :: R :> R ::> Ok
-- foreign import ccall unsafe "ldl_C" zhetrf :: R :> C ::> Ok

foreign import ccall unsafe "ldl_R" ssytrf :: R :> R ::> Ok
foreign import ccall unsafe "ldl_C" chetrf :: R :> C ::> Ok


-- | LDL factorization of a symmetric real matrix, using LAPACK's /dsytrf/.
ldlR :: Matrix Float -> (Matrix Float, [Int])
ldlR = ldlAux ssytrf "ldlR"

-- | LDL factorization of a hermitian complex matrix, using LAPACK's /zhetrf/.
ldlC :: Matrix (Complex Float) -> (Matrix (Complex Float), [Int])
ldlC = ldlAux chetrf "ldlC"

ldlAux f st a = unsafePerformIO $ do
    ldl <- copy ColumnMajor a
    piv <- createVector (rows a)
    (piv #! ldl) f #| st
    return (ldl, map (pred.round) (toList piv))

-----------------------------------------------------------------------------------

-- foreign import ccall unsafe "ldl_S_R" dsytrs :: R ::> R :> R ::> Ok
-- foreign import ccall unsafe "ldl_S_C" zsytrs :: C ::> R :> C ::> Ok

foreign import ccall unsafe "ldl_S_R" ssytrs :: R ::> R :> R ::> Ok
foreign import ccall unsafe "ldl_S_C" csytrs :: C ::> R :> C ::> Ok


-- | Solve a real linear system from a precomputed LDL decomposition ('ldlR'), using LAPACK's /dsytrs/.
ldlsR :: Matrix Float -> [Int] -> Matrix Float -> Matrix Float
ldlsR a piv b = lusAux ssytrs "ldlsR" (fmat a) piv b

-- | Solve a complex linear system from a precomputed LDL decomposition ('ldlC'), using LAPACK's /zsytrs/.
ldlsC :: Matrix (Complex Float) -> [Int] -> Matrix (Complex Float) -> Matrix (Complex Float)
ldlsC a piv b = lusAux csytrs "ldlsC" (fmat a) piv b
