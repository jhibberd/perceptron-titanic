{-
TODO:
-- Add fareGroup
-- Once all weights have been established pick the strongest n and run again
   with just those weights. Might avoid overfitting. Does this improve score?
-}

import Text.CSV
import Debug.Trace
import Data.List

-- Config ----------------------------------------------------------------------

trainingFilename =      "/home/jhibberd/projects/learning/titanic/train.csv"
testFilename =          "/home/jhibberd/projects/learning/titanic/test.csv"
learningRate =          0.0005
trainingIterations =    500


type Filename = String

main = do

    putStrLn "Reading training data from file..."
    dataset <- readCSV trainingFilename True
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
    pprintWeights weights'

    putStrLn "Reading test data from file..."
    testset <- readCSV testFilename True
    putStrLn ("Read " ++ show (length testset) ++ " records.")

    -- Quick n dirty
    {-
    let vectorMatrix' = toVectorMatrix testset
        vectorMatrix'' = combineVectors vectorMatrix'
    let cs = map (classify weights') vectorMatrix''
        cs' = map (show . floor) cs
    prependFile testFilename ("survived":cs')
    -}

pprintWeights :: [Float] -> IO ()
pprintWeights ws = mapM_ f (zip vectorLabels ws)
    where f (lbl, w) = putStrLn ("\t" ++ lbl ++ "\t" ++ show w)

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
    "Is aged <18................",
    "Is aged 18-29..............",
    "Is aged 30-39..............",
    "Is aged 40-49..............",
    "Is aged 50+................",
    "Did embark at 'C'..........",
    "Did embark at 'Q'..........",
    "Did embark at Southampton..",
    "In cabin tier A............",
    "In cabin tier B............",
    "In cabin tier C............",
    "In cabin tier D............",
    "In cabin tier E............",
    "In cabin tier F............",
    "In cabin tier G............",
    "In cabin tier H............"
    "Has no siblings............",
    "Has 1 sibling..............",
    "Has 2 siblings.............",
    "Has 3 siblings.............",
    "Has 4+ siblings............",
    "Has no parents.............",
    "Has 1 parent...............",
    "Has 2 parents..............",
    "Has 3 parents..............",
    "Has 4 parents.............."
    ]

-- | Transform raw, string-based training rows to binary vectors.
--
-- Each value in the binary vector represents whether the row belongs to a 
-- particular set (0/no, 1/yes).
toVectorMatrix :: [[String]]    -- String-based dataset to convert
               -> [[Float]]     -- List of vectors
toVectorMatrix d = [

    -- Gender
    toVectors d 2 (\x -> b (x == "female")),
    toVectors d 2 (\x -> b (x == "male")),

    -- Class
    toVectors d 0 (\x -> b (x == "1")),
    toVectors d 0 (\x -> b (x == "2")),
    toVectors d 0 (\x -> b (x == "3")),
    
    -- Age
    toVectors d 3 (\x -> b ((ageGroup x) == 1)),
    toVectors d 3 (\x -> b ((ageGroup x) == 2)),
    toVectors d 3 (\x -> b ((ageGroup x) == 3)),
    toVectors d 3 (\x -> b ((ageGroup x) == 4)),
    toVectors d 3 (\x -> b ((ageGroup x) == 5)),

    -- Embarked
    toVectors d 9 (\x -> b (x == "C")),
    toVectors d 9 (\x -> b (x == "Q")),
    toVectors d 9 (\x -> b (x == "S")),

    -- Cabin group
    toVectors d 8 (\x -> b ((cabinGroup x) == 1)),
    toVectors d 8 (\x -> b ((cabinGroup x) == 2)),
    toVectors d 8 (\x -> b ((cabinGroup x) == 3)),
    toVectors d 8 (\x -> b ((cabinGroup x) == 4)),
    toVectors d 8 (\x -> b ((cabinGroup x) == 5)),
    toVectors d 8 (\x -> b ((cabinGroup x) == 6)),
    toVectors d 8 (\x -> b ((cabinGroup x) == 7)),
    toVectors d 8 (\x -> b ((cabinGroup x) == 8))

    -- Siblings
    toVectors d 4 (\x -> b ((siblingGroup x) == 1)),
    toVectors d 4 (\x -> b ((siblingGroup x) == 2)),
    toVectors d 4 (\x -> b ((siblingGroup x) == 3)),
    toVectors d 4 (\x -> b ((siblingGroup x) == 4)),
    toVectors d 4 (\x -> b ((siblingGroup x) == 5)),

    -- Parents
    toVectors d 5 (\x -> b (x == "0")),
    toVectors d 5 (\x -> b (x == "1")),
    toVectors d 5 (\x -> b (x == "2")),
    toVectors d 5 (\x -> b (x == "3")),
    toVectors d 5 (\x -> b (x == "4"))

    ]
    where b True =  1
          b False = 0


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
              | n < 18 = 1
              | n < 30 = 2
              | n < 40 = 3
              | n < 50 = 4
              | otherwise = 5

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


-- Helpers ---------------------------------------------------------------------

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
toVectors :: [[String]]         -- Raw dataset
          -> Int                -- Index of column to convert to vector
          -> (String -> Float)  -- Function to convert raw value to vector float.
          -> [Float]
toVectors ds i f = map f onlyCol 
    where onlyCol = map (!!i) ds

-- | Read the contents of a CSV file as a list of lists.
--
-- If 'headless' is set to true then the first line containing the column
-- names is ignored. Any trailing empty line is ignored.
readCSV :: Filename 
        -> Bool 
        -> IO [[String]]
readCSV fn headless = do
    content <- parseCSVFromFile fn
    case content of
        (Left a) -> error ""
        (Right a) -> let a' = removeBlankLines a in
                     return $ if headless then tail a' else a'
    where removeBlankLines = filter (/= [""])

prependFile :: FilePath 
            -> [String] 
            -> IO ()
prependFile fn xs = do
    content <- readFile fn
    let content' = lines content
    let result = map (\(x, y) -> x ++ "," ++ y) $ zip xs content'
    mapM_ putStrLn result

type Sample = ([Float], Float)
type TrainingSet = [Sample]


-- Perceptron algorithm --------------------------------------------------------

-- | Return a list of equal size to the input vectors containing zeros.
initWeights :: [[Float]]    -- All input vectors
            -> [Float]      -- Returns zero'd list of weights of equal length
                            -- to input vectors
initWeights (x:_) = take (length x) $ repeat 0.0

-- | May not produce the best outcome. Maybe pick lowest error score after
-- n iterations?
train :: [Float]            -- Initial or current weights 
      -> TrainingSet 
      -> ([Float], Float)   -- Best error score achieved and corresponding 
                            -- weights
      -> Float              -- Iterations remaining
      -> ([Float], Float)   -- Return best weights and error
train _ _ (bWeights, bError) 0 = (bWeights, bError)
train ws xs (bWeights, bError) n = 

    let (ws', errorSum) = iterate' ws xs 0
        error = errorSum / numTrainingVectors

        newBest = if error < bError
            then (ws', error)
            else (bWeights, bError)

    in train ws' xs newBest (n-1)
    where numTrainingVectors = fromIntegral $ length xs


iterate' :: [Float] 
         -> TrainingSet 
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

