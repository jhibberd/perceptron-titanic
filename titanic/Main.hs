{-
TODO:
-- Try another submission.
-- Still not clear whether we can just drop certain weights with impunity or
   whether they all somehow depend on one another and by dropping some, the
   others become invalid.
-- Try top 25 weights.
-- Try dynamic columns (children * parents) especially (gender, class and age).
-- Cleanup, document, include kaggle links and training/test datasets.
-}

import Text.CSV
import Text.Printf (printf)
import Debug.Trace
import Data.List

-- Config ----------------------------------------------------------------------

trainingFilename =      "/home/jhibberd/projects/learning/titanic/train.csv"
testFilename =          "/home/jhibberd/projects/learning/titanic/test.csv"
learningRate =          0.00005
trainingIterations =    500
numTopFeatures =        20


main = do

    putStrLn "Reading training data from file..."
    dataset <- readCSV trainingFilename
    putStrLn ("Read " ++ show (length dataset) ++ " records.")

    putStrLn "Extracting desired outputs..."
    let outputs = toOutputs dataset
        dataset' = removeOutputs dataset

    putStrLn "Converting dataset to vectors..."
    let vectorMatrix = toVectorMatrix dataset'

    putStrLn ("Generated " ++ show (length vectorMatrix) ++ " vector series.")
    putStrLn "Combining vector series..."
    let combined = combineVectors vectorMatrix
    let combined' = zip combined outputs

    putStrLn "Learning correlation weights..."
    let weights = initWeights combined
    let (weights', error) = train weights combined' ([], 1) trainingIterations
    putStrLn ("Learnt weights with error " ++ show error ++ ":")
    pprintWeights weights' vectorLabels

    putStrLn "Identifying strongest correlations..."
    let bws = bestWeights numTopFeatures weights'
        bestVectorMatrix = filterByIndices vectorMatrix bws 
        bestVectorMatrix' = combineVectors bestVectorMatrix
        bestVectorMatrix'' = zip bestVectorMatrix' outputs
    pprintTopFeatures bws

    putStrLn "Reading test data from file..."
    testset <- readCSV testFilename
    putStrLn ("Read " ++ show (length testset) ++ " records.")

    -- Quick n dirty
    {-
    let vectorMatrix' = toVectorMatrix testset
        vectorMatrix'' = filterByIndices vectorMatrix' bws
        vectorMatrix''' = combineVectors vectorMatrix''
        weights'' = filterByIndices weights' bws

    let cs = map (classify weights'') vectorMatrix'''
        cs' = map (show . floor) cs
    prependFile testFilename ("survived":cs')
    -}


-- | Transform raw, string-based training rows to binary vectors.
--
-- Each value in the binary vector represents whether the row belongs to a 
-- particular set (0/no, 1/yes).
toVectorMatrix :: [[String]]    -- String-based dataset to convert
               -> [[Float]]     -- List of vectors
toVectorMatrix d = [

    -- Gender
    toVectors d (\x -> [x!!2]) (\(x:[]) -> b (x == "female")),
    toVectors d (\x -> [x!!2]) (\(x:[]) -> b (x == "male")),

    -- Class
    toVectors d (\x -> [x!!0]) (\(x:[]) -> b (x == "1")),
    toVectors d (\x -> [x!!0]) (\(x:[]) -> b (x == "2")),
    toVectors d (\x -> [x!!0]) (\(x:[]) -> b (x == "3")),
   
    -- Class and gender
    toVectors d (\x -> [x!!0, x!!2]) (\(c:g:[]) -> b (c == "1" && g == "female")),
    toVectors d (\x -> [x!!0, x!!2]) (\(c:g:[]) -> b (c == "2" && g == "female")),
    toVectors d (\x -> [x!!0, x!!2]) (\(c:g:[]) -> b (c == "3" && g == "female")),
    toVectors d (\x -> [x!!0, x!!2]) (\(c:g:[]) -> b (c == "1" && g == "male")),
    toVectors d (\x -> [x!!0, x!!2]) (\(c:g:[]) -> b (c == "2" && g == "male")),
    toVectors d (\x -> [x!!0, x!!2]) (\(c:g:[]) -> b (c == "3" && g == "male")),

    -- Age
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 1)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 2)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 3)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 4)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 5)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 6)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 7)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 8)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 9)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 10)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 11)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 12)),
    toVectors d (\x -> [x!!3]) (\(x:[]) -> b ((ageGroup x) == 13)),

    -- Embarked
    toVectors d (\x -> [x!!9]) (\(x:[]) -> b (x == "C")),
    toVectors d (\x -> [x!!9]) (\(x:[]) -> b (x == "Q")),
    toVectors d (\x -> [x!!9]) (\(x:[]) -> b (x == "S")),

    -- Cabin group
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 1)),
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 2)),
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 3)),
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 4)),
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 5)),
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 6)),
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 7)),
    toVectors d (\x -> [x!!8]) (\(x:[]) -> b ((cabinGroup x) == 8)),

    -- Siblings
    toVectors d (\x -> [x!!4]) (\(x:[]) -> b ((siblingGroup x) == 1)),
    toVectors d (\x -> [x!!4]) (\(x:[]) -> b ((siblingGroup x) == 2)),
    toVectors d (\x -> [x!!4]) (\(x:[]) -> b ((siblingGroup x) == 3)),
    toVectors d (\x -> [x!!4]) (\(x:[]) -> b ((siblingGroup x) == 4)),
    toVectors d (\x -> [x!!4]) (\(x:[]) -> b ((siblingGroup x) == 5)),

    -- Parents
    toVectors d (\x -> [x!!5]) (\(x:[]) -> b (x == "0")),
    toVectors d (\x -> [x!!5]) (\(x:[]) -> b (x == "1")),
    toVectors d (\x -> [x!!5]) (\(x:[]) -> b (x == "2")),
    toVectors d (\x -> [x!!5]) (\(x:[]) -> b (x == "3")),
    toVectors d (\x -> [x!!5]) (\(x:[]) -> b (x == "4")),

    -- Fare
    toVectors d (\x -> [x!!7]) (\(x:[]) -> b ((fareGroup x) == 1)),
    toVectors d (\x -> [x!!7]) (\(x:[]) -> b ((fareGroup x) == 2)),
    toVectors d (\x -> [x!!7]) (\(x:[]) -> b ((fareGroup x) == 3)),
    toVectors d (\x -> [x!!7]) (\(x:[]) -> b ((fareGroup x) == 4)),
    toVectors d (\x -> [x!!7]) (\(x:[]) -> b ((fareGroup x) == 5))

    ]
    where b True =  1
          b False = 0


-- Log -------------------------------------------------------------------------

-- | Pretty print the top features.
pprintTopFeatures :: [Int]  -- Index positions of top weights
                  -> IO ()
pprintTopFeatures is = mapM_ f (filterByIndices vectorLabels is)
    where f lbl = putStrLn ("\t" ++ lbl)

-- | Pretty print a list of features and their respective learnt weights. 
pprintWeights :: [Float]    -- Weights
              -> [String]   -- Labels
              -> IO ()
pprintWeights ws lbls = mapM_ f (zip lbls ws)
    where f (lbl, w) = putStrLn ("\t" ++ lbl ++ "\t" ++ printf "%+.6f" w)

-- | Human friendly labels explaining what each vector represents.
--
-- Used to pretty printing the learnt weights.
vectorLabels :: [String]
vectorLabels = [
    "Is female..................",
    "Is male....................",
    "Is first class.............",
    "Is second class............",
    "Is third class.............",
    "Is first class and female..",
    "Is second class and female.",
    "Is third class and female..",
    "Is first class and male....",
    "Is second class and male...",
    "Is third class and male....",
    "Is aged <5.................",
    "Is aged 5-9................",
    "Is aged 10-14..............",
    "Is aged 15-19..............",
    "Is aged 20-24..............",
    "Is aged 25-29..............",
    "Is aged 30-34..............",
    "Is aged 35-39..............",
    "Is aged 40-44..............",
    "Is aged 45-49..............",
    "Is aged 50-54..............",
    "Is aged 55-59..............",
    "Is aged 60+................",
    "Did embark at Cherbourg....",
    "Did embark at Queenstown...",
    "Did embark at Southampton..",
    "In cabin tier A............",
    "In cabin tier B............",
    "In cabin tier C............",
    "In cabin tier D............",
    "In cabin tier E............",
    "In cabin tier F............",
    "In cabin tier G............",
    "In cabin tier H............",
    "Has no siblings............",
    "Has 1 sibling..............",
    "Has 2 siblings.............",
    "Has 3 siblings.............",
    "Has 4+ siblings............",
    "Has no parents.............",
    "Has 1 parent...............",
    "Has 2 parents..............",
    "Has 3 parents..............",
    "Has 4 parents..............",
    "Fare is < 25...............",
    "Fare is < 50...............",
    "Fare is < 75...............",
    "Fare is < 100..............",
    "Fare is 100+..............."
    ]


-- Vector transformations ------------------------------------------------------

siblingGroup :: String -> Float
siblingGroup "" =   0
siblingGroup "0" =  1
siblingGroup "1" =  2
siblingGroup "2" =  3
siblingGroup "3" =  4 
siblingGroup _ =    5 

ageGroup :: String -> Float
ageGroup "" = 0
ageGroup x = group $ (read x :: Float)
    where group n
              | n < 5 =     1
              | n < 10 =    2
              | n < 15 =    3
              | n < 20 =    4
              | n < 25 =    5
              | n < 30 =    6
              | n < 35 =    7
              | n < 40 =    8
              | n < 45 =    9
              | n < 50 =    10
              | n < 55 =    11
              | n < 60 =    12
              | otherwise = 13

cabinGroup :: String -> Float
cabinGroup "" = 0
cabinGroup x = letterScore cabin
    where cabin = (words x) !! 0
          letterScore c = case (c !! 0) of
                              'A' -> 1
                              'B' -> 2
                              'C' -> 3
                              'D' -> 4
                              'E' -> 5
                              'F' -> 6
                              'G' -> 7
                              'T' -> 8

fareGroup :: String -> Float
fareGroup "" = 0
fareGroup x
    | x' < 25.0 =   1
    | x' < 50.0 =   2
    | x' < 75.0 =   3
    | x' < 100.0 =  4
    | otherwise =   5
    where x' = read x


-- Helpers ---------------------------------------------------------------------

-- | Return only elements from a list whose indices are present in a second
-- list.
filterByIndices :: [a]      -- All available vector matrices.
                -> [Int]    -- List of indices to filter by.
                -> [a]      -- Filtered vector matrices.
filterByIndices xs is = map snd filtered
    where indexed = zip [0..] xs
          filtered = filter (\x -> (fst x) `elem` is) indexed

-- | Given a list of weights return the index positions of the highest weights
-- in absolute terms.
bestWeights :: Int      -- Number of top weights to return
            -> [Float]  -- List of weights from which best are chosen
            -> [Int]    -- Index positions of top weights
bestWeights n ws = take n indexOnly
    where absolute = map abs ws
          indexed = zip [0..] absolute
          sorted = reverse $ sortBy (\x y -> compare (snd x) (snd y)) indexed
          indexOnly = map fst sorted

-- | Combines multiple vector lists.
--
-- Similar to zip but produces lists rather than tuples and works on any number
-- of input vector lists.
--
-- [[1, 0, 1, 1], [0, 1, 1, 1], ...]
-- =>
-- [[1, 0], [0, 1], [1, 1], [1, 1], ...]
combineVectors :: [[Float]] -> [[Float]]
combineVectors xs = combine xs n 0 
    where n = length (xs !! 0)
          combine xs n i
              | i == n = []
              | otherwise = map (!!i) xs : combine xs n (i+1) 

-- | Convert a raw (training) dataset into a list of desired outputs, as
-- contained in the dataset.
toOutputs :: [[String]] -> [Float]
toOutputs = map (read . head)

-- | Remove the outputs from a training set to standardise the format.
removeOutputs :: [[String]] -> [[String]]
removeOutputs = map tail

-- | Convert a list of raw string CSV values to a list of vectors
--
-- [["a", "b", "c"], ["d", "e", "f"], ...] => [[1], [2]]
toVectors :: [[String]]                 -- Raw dataset
          -> ([String] -> [String])     -- Function to extract desired columns
                                        -- from a raw row.
          -> ([String] -> Float)        -- Function to convert a list of raw
                                        -- row columns to a single feature.
          -> [Float]
toVectors ds extract toFeature = map (toFeature . extract) ds

-- | Read the contents of a CSV file as a list of lists.
--
-- Any trailing empty line is ignored.
readCSV :: String   -- File path 
        -> IO [[String]]
readCSV fn = do
    content <- parseCSVFromFile fn
    case content of
        (Left a) -> error ""
        (Right a) -> return . tail $ removeBlankLines a
    where removeBlankLines = filter (/= [""])

-- | Prepend the rows of a file with elements from a list.
prependFile :: String       -- File path 
            -> [String]     -- Elements to prepend
            -> IO ()
prependFile fn xs = do
    content <- readFile fn
    let content' = lines content
        result = map (\(x, y) -> x ++ "," ++ y) $ zip xs content'
    mapM_ putStrLn result


-- Perceptron algorithm --------------------------------------------------------

-- | Return a list of equal size to the input vectors containing zeros.
initWeights :: [[Float]]    -- All input vectors
            -> [Float]      -- Returns zero'd list of weights of equal length
                            -- to input vectors
initWeights (x:_) = take (length x) $ repeat 0.0

-- | Learn a list of weights by repeatedly iterating over a training dataset.
-- The most successful weights are returned after n iterations.
train :: [Float]                -- Initial or current weights 
      -> [([Float], Float) ]    -- Training set
      -> ([Float], Float)       -- Best error score achieved and corresponding 
                                -- weights
      -> Float                  -- Iterations remaining
      -> ([Float], Float)       -- Return best weights and error
train _ _ (bWeights, bError) 0 = (bWeights, bError)
train ws xs (bWeights, bError) n = 

    let (ws', errorSum) = iterate' ws xs 0
        error = errorSum / numTrainingVectors

        newBest = if error < bError
            then (ws', error)
            else (bWeights, bError)

    in train ws' xs newBest (n-1)
    where numTrainingVectors = fromIntegral $ length xs

-- | With a starting list of weights, iterate through the training dataset
-- and adjust the weights by applying the perceptron algorithm whenever items
-- are misclassified.
iterate' :: [Float]             -- Current or initial weights
         -> [([Float], Float)]  -- Training set
         -> Float               -- Error 
         -> ([Float], Float)    -- Weights and error
iterate' ws [] e = (ws, e)
iterate' ws ((x, expected):xs) errorSum = 
    let actual = classify ws x
        error = expected - actual
        errorSum' = errorSum + abs error
        ws' = adjustWeights ws x error
    in iterate' ws' xs errorSum'

-- | Given weights, classify a vector.
classify :: [Float]         -- Weights
         -> [Float]         -- Vector 
         -> Float           -- Classification (1 or 0)
classify ws xs = bin $ sum [w * x | (w, x) <- zip ws xs]
    where bin x 
              | x > 0 = 1
              | otherwise = 0

-- | Given weights, a vector and the error that resulted from classifying that
-- vector with those weights, adjust the weights so as to reduce the chance
-- of future misclassification.
adjustWeights :: [Float]        -- Weights
              -> [Float]        -- Vector
              -> Float          -- Error
              -> [Float]        -- Adjusted weights
adjustWeights ws _ 0 = ws
adjustWeights ws xs e = [w + (learningRate * e * x) | (w, x) <- zip ws xs]

