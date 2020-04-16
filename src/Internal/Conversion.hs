{-# LANGUAGE FlexibleContexts       #-}
{-# LANGUAGE FlexibleInstances      #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses  #-}
{-# LANGUAGE TypeFamilies           #-}
{-# LANGUAGE UndecidableInstances   #-}

-----------------------------------------------------------------------------
-- |
-- Module      :  Numeric.Conversion
-- Copyright   :  (c) Alberto Ruiz 2010
-- License     :  BSD3
-- Maintainer  :  Alberto Ruiz
-- Stability   :  provisional
--
-- Conversion routines
--
-----------------------------------------------------------------------------


module Internal.Conversion (
    Complexable(..), RealElement,
    module Data.Complex
) where

import           Control.Arrow       ((***))
import           Data.Complex
import           Internal.Matrix
import           Internal.Vector
import           Internal.Vectorized

-------------------------------------------------------------------

-- | Supported single-double precision type pairs
class (Element s, Element d) => Precision s d | s -> d, d -> s where
    double2FloatG :: Vector d -> Vector s
    float2FloatG :: Vector s -> Vector d

instance Precision Float Float where
    double2FloatG = double2FloatV
    float2FloatG = float2FloatV

instance Precision (Complex Float) (Complex Float) where
    double2FloatG = asComplex . double2FloatV . asReal
    float2FloatG = asComplex . float2FloatV . asReal

instance Precision I Z where
    double2FloatG = long2intV
    float2FloatG = int2longV


-- | Supported real types
class (Element t, Element (Complex t), RealFloat t)
    => RealElement t

-- instance RealElement Float
instance RealElement Float


-- | Structures that may contain complex numbers
class Complexable c where
    toComplex'   :: (RealElement e) => (c e, c e) -> c (Complex e)
    fromComplex' :: (RealElement e) => c (Complex e) -> (c e, c e)
    comp'        :: (RealElement e) => c e -> c (Complex e)
    single'      :: Precision a b => c b -> c a
    double'      :: Precision a b => c a -> c b


instance Complexable Vector where
    toComplex' = toComplexV
    fromComplex' = fromComplexV
    comp' v = toComplex' (v,constantD 0 (dim v))
    single' = double2FloatG
    double' = float2FloatG


-- | creates a complex vector from vectors with real and imaginary parts
toComplexV :: (RealElement a) => (Vector a, Vector a) ->  Vector (Complex a)
toComplexV (r,i) = asComplex $ flatten $ fromColumns [r,i]

-- | the inverse of 'toComplex'
fromComplexV :: (RealElement a) => Vector (Complex a) -> (Vector a, Vector a)
fromComplexV z = (r,i) where
    [r,i] = toColumns $ reshape 2 $ asReal z


instance Complexable Matrix where
    toComplex' = uncurry $ liftMatrix2 $ curry toComplex'
    fromComplex' z = (reshape c *** reshape c) . fromComplex' . flatten $ z
        where c = cols z
    comp' = liftMatrix comp'
    single' = liftMatrix single'
    double' = liftMatrix double'

