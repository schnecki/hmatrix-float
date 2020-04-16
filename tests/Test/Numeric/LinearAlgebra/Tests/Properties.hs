{-# LANGUAGE CPP              #-}
{-# LANGUAGE DataKinds        #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies     #-}

{-# OPTIONS_GHC -fno-warn-missing-signatures #-}

-----------------------------------------------------------------------------
{- |
Module      :  Numeric.LinearAlgebra.Tests.Properties
Copyright   :  (c) Alberto Ruiz 2008
License     :  BSD3
Maintainer  :  Alberto Ruiz
Stability   :  provisional

Testing properties.

-}

module Test.Numeric.LinearAlgebra.Tests.Properties (
    dist, (|~|), (~~), (~:), Aprox((:~)), (~=),
    zeros, ones,
    square,
    unitary,
    hermitian,
    wellCond,
    positiveDefinite,
    upperTriang,
    upperHessenberg,
    luProp,
    invProp,
    pinvProp,
    detProp,
    nullspaceProp,
--    bugProp,
    svdProp1, svdProp1a, svdProp1b, svdProp2, svdProp3, svdProp4,
    svdProp5a, svdProp5b, svdProp6a, svdProp6b, svdProp7,
    eigProp, eigSHProp, eigProp2, eigSHProp2,
    geigProp,
    qrProp, rqProp, rqProp1, rqProp2, rqProp3,
    hessProp,
    schurProp1, schurProp2,
    cholProp, exactProp,
    expmDiagProp,
    multProp1, multProp2,
    subProp,
    linearSolveProp, linearSolvePropH, linearSolveProp2,

    -- Binary properties
    vectorBinaryRoundtripProp
  , staticVectorBinaryRoundtripProp
  , matrixBinaryRoundtripProp
  , staticMatrixBinaryRoundtripProp
  , staticVectorBinaryFailProp
) where

import           Numeric.LinearAlgebra.HMatrix hiding (Testable)
import qualified Numeric.LinearAlgebra.Static  as Static
import           Test.QuickCheck

import           Data.Binary
import           Data.Binary.Get               (runGet)
import           Data.Either                   (isLeft)
#if MIN_VERSION_base(4,11,0)
import           Prelude                       hiding ((<>))
#endif

import           Debug.Trace

(~=) :: Float -> Float -> Bool
a ~= b = abs (a - b) < 1e-10

trivial :: Testable a => Bool -> a -> Property
trivial = (`classify` "trivial")

-- relative error
dist :: (Num a, Normed a) => a -> a -> Float
dist = relativeError norm_Inf

infixl 4 |~|
a |~| b = a :~1~: b
--a |~| b = dist a b < 10^^(-10)

a ~~ b = fromList a |~| fromList b

data Aprox a = (:~) a Int
-- (~:) :: (Normed a, Num a) => Aprox a -> a -> Bool
a :~n~: b = dist a b < 10^^(-n)

------------------------------------------------------

square m = rows m == cols m

-- orthonormal columns
orthonormal m = tr m <> m |~| ident (cols m)

unitary m = square m && orthonormal m

hermitian m = square m && m |~| tr m

wellCond m = rcond m > 1/100

positiveDefinite m = minimum (toList e) > 0
    where (e,_v) = eigSH m

upperTriang m = rows m == 1 || down == z
    where down = fromList $ concat $ zipWith drop [1..] (toLists (tr m))
          z = konst 0 (size down)

upperHessenberg m = rows m < 3 || down == z
    where down = fromList $ concat $ zipWith drop [2..] (toLists (tr m))
          z = konst 0 (size down)

zeros (r,c) = reshape c (konst 0 (r*c))

ones (r,c) = zeros (r,c) + 1

-----------------------------------------------------

luProp m =
  -- trace ("x: " ++ show m)
  -- trace ("m: " ++ show m ++ " =~ " ++ show (p <> l <> u) ++ " ... " ++ show (m |~| p <> l <> u))
  -- trace ("f (det p) |~| f s: " ++ show (f (det p)) ++ " = " ++ show  (f s) ++ " ... " ++ show (f (det p) |~| f s))
  m |~| p <> l <> u && f (det p) |~| f s
    where (l,u,p,s) = lu m
          f x = fromList [x]

invProp m = m <> inv m |~| ident (rows m)

pinvProp m =  m <> p <> m |~| m
           && p <> m <> p |~| p
           && hermitian (m<>p)
           && hermitian (p<>m)
    where p = pinv m

detProp m =
  trace ("d1: " ++ show (s d1, s d2))
  s d1 |~| s d2
    where d1 = det m
          d2 = det' * det q
          det' = product $ toList $ takeDiag r
          (q,r) = qr m
          s x = fromList [x]

nullspaceProp m = null nl `trivial` (null nl || m <> n |~| zeros (r,c)
                                     && orthonormal n)
    where n = nullspaceSVD (Left (1*peps)) m (rightSV m)
          nl = toColumns n
          r = rows m
          c = cols m - rank m

------------------------------------------------------------------
{-
-- testcase for nonempty fpu stack
-- uncommenting unitary' signature eliminates the problem
bugProp m = m |~| u <> real d <> tr v && unitary' u && unitary' v
    where (u,d,v) = svd m
          -- unitary' :: (Num (Vector t), Field t) => Matrix t -> Bool
          unitary' a = unitary a
-}
------------------------------------------------------------------

-- fullSVD
svdProp1 m = m |~| u <> real d <> tr v && unitary u && unitary v
  where
    (u,s,v) = svd m
    d = diagRect 0 s (rows m) (cols m)

svdProp1a svdfun m = m |~| u <> real d <> tr v && unitary u && unitary v
  where
    (u,s,v) = svdfun m
    d = diagRect 0 s (rows m) (cols m)

svdProp1b svdfun m = unitary u && unitary v
  where
    (u,_,v) = svdfun m

-- thinSVD
svdProp2 thinSVDfun m
    =  m |~| u <> diag (real s) <> tr v
    && orthonormal u && orthonormal v
    && size s == min (rows m) (cols m)
  where
    (u,s,v) = thinSVDfun m

-- compactSVD
svdProp3 m = (m |~| u <> real (diag s) <> tr v
             && orthonormal u && orthonormal v)
  where
    (u,s,v) = compactSVD m

svdProp4 m' = m |~| u <> real (diag s) <> tr v
           && orthonormal u && orthonormal v
           && (size s == r || r == 0 && size s == 1)
  where
    (u,s,v) = compactSVD m
    m = fromBlocks [[m'],[m']]
    r = rank m'

svdProp5a m = all (s1|~|) [s3,s5] where
    s1       = singularValues (m :: Matrix Float)
--  s2       = svRd m
    (_,s3,_) = svd m
--  (_,s4,_) = svdRd m
    (_,s5,_) = thinSVD m
--  (_,s6,_) = thinSVDRd m

svdProp5b m = all (s1|~|) [s3,s5] where
    s1       = singularValues (m :: Matrix (Complex Float))
--  s2       = svCd m
    (_,s3,_) = svd m
--  (_,s4,_) = svdCd m
    (_,s5,_) = thinSVD m
--  (_,s6,_) = thinSVDCd m

svdProp6a m = s |~| s' && v |~| v' && s |~| s'' && u |~| u'
  where
    (u,s,v) = svd (m :: Matrix Float)
    (s',v') = rightSV m
    (u',s'') = leftSV m

svdProp6b m = s |~| s' && v |~| v' && s |~| s'' && u |~| u'
  where
    (u,s,v) = svd (m :: Matrix (Complex Float))
    (s',v') = rightSV m
    (u',s'') = leftSV m

svdProp7 m = s |~| s' && u |~| u' && v |~| v' && s |~| s'''
  where
    (u,s,v) = svd m
    (s',v') = rightSV m
    (u',_s'') = leftSV m
    s''' = singularValues m

------------------------------------------------------------------

eigProp m = complex m <> v |~| v <> diag s
    where (s, v) = eig m

eigSHProp m = m <> v |~| v <> real (diag s)
              && unitary v
              && m |~| v <> real (diag s) <> tr v
    where (s, v) = eigSH' m

eigProp2 m = fst (eig m) |~| eigenvalues m

eigSHProp2 m = fst (eigSH' m) |~| eigenvaluesSH' m

geigProp a b = a' <> v <> diag betas' |~| b' <> v <> diag alphas
    where (alphas, betas, v) = geig a b
          betas' = complex betas
          a' = complex a
          b' = complex b

------------------------------------------------------------------

qrProp m = q <> r |~| m && unitary q && upperTriang r
    where (q,r) = qr m

rqProp m = r <> q |~| m && unitary q && upperTriang' r
    where (r,q) = rq m

rqProp1 m = r <> q |~| m
    where (r,q) = rq m

rqProp2 m = unitary q
    where (_r,q) = rq m

rqProp3 m = upperTriang' r
    where (r,_q) = rq m

upperTriang' r = upptr (rows r) (cols r) * r |~| r
    where upptr f c = build (f,c) $ \r' c' -> if r'-t > c' then 0 else 1
              where t = fromIntegral (f-c)

hessProp m = m |~| p <> h <> tr p && unitary p && upperHessenberg h
    where (p,h) = hess m

schurProp1 m = m |~| u <> s <> tr u && unitary u && upperTriang s
    where (u,s) = schur m

schurProp2 m = m |~| u <> s <> tr u && unitary u && upperHessenberg s -- fixme
    where (u,s) = schur m

cholProp m = m |~| tr c <> c && upperTriang c
    where c = chol (trustSym m)

exactProp m = chol (trustSym m) == chol (trustSym (m+0))

expmDiagProp m = expm (logm m) :~ 7 ~: complex m
    where logm = matFunc log

-- reference multiply
mulH a b = fromLists [[ doth ai bj | bj <- toColumns b] | ai <- toRows a ]
    where doth u v = sum $ zipWith (*) (toList u) (toList v)

multProp1 p (a,b) = (a <> b) :~p~: (mulH a b)

multProp2 p (a,b) = (tr (a <> b)) :~p~: (tr b <> tr a)

linearSolveProp f m = f m m |~| ident (rows m)

linearSolvePropH f m = f m (unSym m) |~| ident (rows (unSym m))

linearSolveProp2 f (a,x) = not wc `trivial` (not wc || a <> f a b |~| b)
    where q = min (rows a) (cols a)
          b = a <> x
          wc = rank a == q

subProp m = m == (conj . tr . fromColumns . toRows) m

------------------------------------------------------------------

vectorBinaryRoundtripProp :: Vector Float -> Bool
vectorBinaryRoundtripProp vec = decode (encode vec) == vec

staticVectorBinaryRoundtripProp :: Static.R 5 -> Bool
staticVectorBinaryRoundtripProp vec =
  let
    decoded = decode (encode vec) :: Static.R 500
  in
    Static.extract decoded == Static.extract vec

matrixBinaryRoundtripProp :: Matrix Float -> Bool
matrixBinaryRoundtripProp mat = decode (encode mat) == mat

staticMatrixBinaryRoundtripProp :: Static.L 100 200 -> Bool
staticMatrixBinaryRoundtripProp mat =
  let
    decoded = decode (encode mat) :: Static.L 100 200
  in
    (Static.extract decoded) == (Static.extract mat)

staticVectorBinaryFailProp :: Static.R 20 -> Bool
staticVectorBinaryFailProp vec =
  let
    decoded = runGet get (encode vec) :: Either String (Static.R 50)
  in
    isLeft decoded
