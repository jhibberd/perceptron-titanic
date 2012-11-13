
import Text.CSV
import Debug.Trace
import Data.List

trainingFilename = "/home/jhibberd/projects/learning/titanic/train.csv"
testFilename = "/home/jhibberd/projects/learning/titanic/test.csv"

threshold = 0.449
bias = 0.0
learningRate = 0.0005

type Filename = String

main = do

    --putStrLn "Reading training data from file..."
    dataset <- readCSV trainingFilename True
    --putStrLn ("Read " ++ show (length dataset) ++ " records.")

    --putStrLn "Extracting desired outputs..."
    let outputs = toOutputs dataset
        dataset' = removeOutputs dataset

    --putStrLn "Converting dataset to vectors..."
    let vectorMatrix = toVectorMatrix dataset'

    --putStrLn ("Generated " ++ show (length vectorMatrix) ++ " vector series.")
    --putStrLn "Combining vector series..."
    let combined = combineVectors vectorMatrix
    let combined' = zip combined outputs

    --putStrLn "Learning correlation weights..."
    let weights = initWeights combined
    let weights' = train weights combined'
    --putStrLn "Learnt weights:"
    --putStrLn $ show weights'


    --putStrLn "Reading test data from file..."
    testset <- readCSV testFilename True
    --putStrLn ("Read " ++ show (length testset) ++ " records.")

    -- Quick n dirty

    let vectorMatrix' = toVectorMatrix testset
        vectorMatrix'' = combineVectors vectorMatrix'
    let cs = map (classify weights') vectorMatrix''
        cs' = map (show . floor) cs
    prependFile testFilename ("survived":cs')


    {-
    -- Learn weights
    trainingCSV <- parseCSVFromFile trainingFilename
    let dataSet = getDataSet trainingCSV
        trainingOutcomes = getOutcomesFromDataSet dataSet
        trainingItems = getItemsFromDataSet dataSet
        trainingSet = zip (vectoriseItems trainingItems) trainingOutcomes
        learntWeights = train (initWeights trainingItems) trainingSet

    let learntWeights' = traceShow learntWeights learntWeights

    -- Apply learnt weights to test set
    testCSV <- parseCSVFromFile testFilename
    let testSet = getTestSet testCSV

    let cs = map (classify learntWeights') (vectoriseItems testSet)
        cs' = map (show . floor) cs
    prependFile testFilename ("survived":cs')
    -}

toVectorMatrix :: [[String]] -> [[Float]] 
toVectorMatrix ds = [

    toVectors ds 2 (\x -> if x == "female" then 1 else 0),
    toVectors ds 2 (\x -> if x == "male" then 1 else 0),

    toVectors ds 0 (\x -> if x == "1" then 1 else 0),
    toVectors ds 0 (\x -> if x == "2" then 1 else 0),
    toVectors ds 0 (\x -> if x == "3" then 1 else 0),

    toVectors ds 3 (isAgeGroup 1),
    toVectors ds 3 (isAgeGroup 2),
    toVectors ds 3 (isAgeGroup 3),
    toVectors ds 3 (isAgeGroup 4),
    toVectors ds 3 (isAgeGroup 5)

    ]

isAgeGroup :: Float -> String -> Float
isAgeGroup g a = if ageGroup a == g then 1 else 0

ageGroup :: String -> Float
ageGroup "" = 0
ageGroup x = group $ (read x :: Float)
    where group n
              | n < 18 = 1
              | n < 30 = 2
              | n < 40 = 3
              | n < 50 = 4
              | otherwise = 5

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
readCSV :: Filename -> Bool -> IO [[String]]
readCSV fn headless = do
    content <- parseCSVFromFile fn
    case content of
        (Left a) -> error ""
        (Right a) -> let a' = removeBlankLines a in
                     return $ if headless then tail a' else a'
    where removeBlankLines = filter (/= [""])


-- Perceptron Algorithm --------------------------------------------------------

-- | Return a list of equal size to the input vectors containing zeros.
initWeights :: [[Float]] -> Weights
initWeights (x:_) = take (length x) $ repeat 0.0

{-
-- Original 'train' function whos exit condition was an error sum below a
-- constant threshold
train :: Weights -> TrainingSet -> Weights
train ws xs = let (ws', errorSum) = iterate' ws xs 0
              in if errorSum / n <= 0-- threshold
                  then ws'
                  else train ws' (traceShow (errorSum / n)  xs)
    where n = fromIntegral $ length xs
-}

train :: Weights -> TrainingSet -> Weights
train ws xs = let (ws', errorSum) = iterate' ws xs 0
                  errorSum' = errorSum / n
              in if ws == ws' -- End when weights-in == weights following iteration. Otherwise infinite loop.
                  then ws'
                  else train ws' xs
    where n = fromIntegral $ length xs

iterate' :: Weights -> TrainingSet -> ErrorSum -> (Weights, ErrorSum)
iterate' ws [] e = (ws, e)
iterate' ws ((x, expected):xs) errorSum = 
    let actual = classify ws x
        error = expected - actual
        errorSum' = errorSum + abs error
        ws' = adjustWeights ws x error
    in iterate' ws' xs errorSum'

classify :: Weights -> InputVector -> Classification
classify ws xs = bin $ sum [w * x | (w, x) <- zip ws xs]
    where bin x | x > 0 = 1
          bin x | otherwise = 0

adjustWeights :: Weights -> InputVector -> Error -> Weights
adjustWeights ws _ 0 = ws
adjustWeights ws xs e = [w + (learningRate * e * x) | (w, x) <- zip ws xs]



-- OLD -------------------------------------------------------------------------

-- Usage: getStrongestSeq [1, (-6), 2, 3, 4, 5] (permutations [1, 2, 3])

-- | Given a list of weights and a list of sequences that those weights
-- correspond to, the function will return a pair containing the largest
-- absolute weight and its associated sequence.
getStrongestSeq :: [Float] -> [[a]] -> ([a], Float)
getStrongestSeq ns xs = let (i, x) = topEl in (xs !! i, x) 
    where topEl = foldr combiningFunc (0, 0) indexed
          indexed = zip [0..] ns -- indexed elements in 'ns' list
          combiningFunc (i, x) (topI, topX) 
              | abs x > topX = (i, abs x)
              | otherwise = (topI, topX)

{-
getStrongestPermutation :: IO ()
getStrongestPermutation = do

    trainingCSV <- parseCSVFromFile trainingFilename
    let dataSet = getDataSet trainingCSV
        trainingOutcomes = getOutcomesFromDataSet dataSet
        trainingItems = getItemsFromDataSet dataSet
        trainingSet = zip (vectoriseItems trainingItems) trainingOutcomes
        learntWeights = train (initWeights trainingItems) trainingSet
        max = maximum $ map abs learntWeights
        i = elemIndex
    return 3
-}

-- | Return the index of the element with the absolute max value.
--absMaxIndex :: [Float] -> Int
--absMaxIndex xs = foldr (\(i, x), (biggestI, biggestX) -> if abs x > biggestX then (i, abs x) else (biggestI, biggestX)) 0 indexedXs
--    where indexedXs = zip [0...] xs

getOutcomesFromDataSet :: [[String]] -> [Float]
getOutcomesFromDataSet = map (asBinary . head)
    where asBinary "0" = 0
          asBinary "1" = 1

getItemsFromDataSet :: [[String]] -> [[String]]
getItemsFromDataSet = map tail

--vectoriseItems :: [[String]] -> [[Float]]
--vectoriseItems = map toVector

-- | Read the test file as a list of lists.
--getTestSet :: Either a [[String]] -> [[String]]
--getTestSet (Left a) = error "Can't read test file."
--getTestSet (Right a) = tail $ init a

prependFile :: FilePath -> [String] -> IO ()
prependFile fn xs = do
    content <- readFile fn
    let content' = lines content
    let result = map (\(x, y) -> x ++ "," ++ y) $ zip xs content'
    mapM_ putStrLn result

type Weights = [Float]
type Sample = (InputVector, Float)
type TrainingSet = [Sample]
type InputVector = [Float]
type Error = Float
type ErrorSum = Float
type Classification = Float

-- TODO(jhibberd) Independent function to test which permutation has strongest
-- correlation with survival rate. Use that permutation only. Suspect too many
-- weights causes watering down of results where each outcome is barely above
-- or below the threshold.

-- | To vector -----------------------------------------------------------------

-- | Convert an items of string types to a vector of float values.
--
-- TODO(jhibberd) List all fields in items.
-- TODO(jhibberd) Just don't include mapping for those that can't be 
-- interpreted.
-- TODO(jhibberd) Max 5 permutations, so formatter which converts actual value
-- to a category that appears in the permutation
--toVector :: [String] -> [Float]
--toVector xs = map (\(i, f) -> f (xs !! i)) converters
--    where converters = (permMap 3 ageGroup ("":(map show [1..5])))
    {-
    where converters = [ mapToFloat ["1", "2", "3"] id ] ++
                       [(1, quantifyName)] ++
                       (permMap 2 id ["male", "female"]) ++ 
                       (permMap 3 ageGroup ("":(map show [1..5]))) ++
                       [ (3, quantifyAge) ] ++
                       (permMap 4 sibsp (map show [0..4])) ++
                       (permMap 5 sibsp (map show [0..4])) ++
                       [
                         (6, quantifyTicket)
                       , (7, quantifyFare)
                       , (8, quantifyCabin)
                       , (9, quantifyEmbarked)
                       ]
    -}

-- Apparently gender makes no difference.

{-
ageGroup :: String -> String
ageGroup "" = ""
ageGroup x = show $ ((floor ((read x) / 20)) + 1)
-}
sibsp :: String -> String
sibsp x
    | (read x) > 3 = "4"
    | otherwise = x

permMap :: Int -> (String -> String) -> [String] -> [(Int, (String -> Float))]
permMap i f m = map (\x -> (i, mapToFloat x f)) $ permutations m

mapToFloat :: [String] -> (String -> String) -> String -> Float
mapToFloat xs f x = case elemIndex (f x) xs of
                        Just i -> fromIntegral (i+1)
                        Nothing -> error "Parse error"

-- | Virtually impossible to quantify name into a meaninful number.
quantifyName :: String -> Float
quantifyName x = 0

quantifyAge :: String -> Float
quantifyAge "" = 0
quantifyAge x = read x

quantifySibsp :: String -> Float
quantifySibsp = read

quantifyParch :: String -> Float
quantifyParch = read

-- | Ignore for now. It may be that the ticket number can be deconstructed to
-- give an indication of where on the ship the passenger was most likely to
-- have been at the time of the accident.
quantifyTicket :: String -> Float
quantifyTicket x = 0

quantifyFare :: String -> Float
quantifyFare "" = 0
quantifyFare x = read x

-- | TODO(jhibberd) Try "dynamic fields" with lists ordered in multiple ways

-- | Ignore for now. See 'quantifyTicket'.
quantifyCabin :: String -> Float
quantifyCabin "" = 0
quantifyCabin x = letterScore cabin * number tail'
    where cabin = (words x) !! 0
          tail' = tail cabin
          number n 
              | n == "" = 0
              | otherwise = read n / 100
          letterScore c = case (c !! 0) of
                              'A' -> 1
                              'B' -> 2
                              'C' -> 3
                              'D' -> 4
                              'E' -> 5
                              'F' -> 6
                              'G' -> 7
                              'T' -> 8

quantifyEmbarked :: String -> Float
quantifyEmbarked "" = 0
quantifyEmbarked "C" = 1
quantifyEmbarked "Q" = 2
quantifyEmbarked "S" = 3

