
import Text.CSV
import Debug.Trace
import Data.List

trainingFilename = "/home/jhibberd/projects/learning/titanic/train.csv"
testFilename = "/home/jhibberd/projects/learning/titanic/test.csv"

threshold = 0.3 --0.265
bias = 0.5
learningRate = 0.0001

main = do

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

getOutcomesFromDataSet :: [[String]] -> [Float]
getOutcomesFromDataSet = map (asBinary . head)
    where asBinary "0" = 0
          asBinary "1" = 1

getItemsFromDataSet :: [[String]] -> [[String]]
getItemsFromDataSet = map tail

vectoriseItems :: [[String]] -> [[Float]]
vectoriseItems = map toVector

-- | Read the training file as a list of lists.
getDataSet :: Either a [[String]] -> [[String]]
getDataSet (Left a) = error "Can't read training file."
getDataSet (Right a) = stripHeadAndFoot a

-- | Read the test file as a list of lists.
getTestSet :: Either a [[String]] -> [[String]]
getTestSet (Left a) = error "Can't read test file."
getTestSet (Right a) = tail $ init a

-- | Return a list of equal size to the input vectors containing zeros.
initWeights :: [[String]] -> Weights
initWeights (v:_) = take (length $ toVector v) $ repeat 0.0

prependFile :: FilePath -> [String] -> IO ()
prependFile fn xs = do
    content <- readFile fn
    let content' = lines content
    let result = map (\(x, y) -> x ++ "," ++ y) $ zip xs content'
    mapM_ putStrLn result

-- | Remove headers from empty line from CSV file.
stripHeadAndFoot :: [a] -> [a]
stripHeadAndFoot = tail . init

type Weights = [Float]
type Sample = (InputVector, Float)
type TrainingSet = [Sample]
type InputVector = [Float]
type Error = Float
type ErrorSum = Float
type Classification = Float

-- | Perceptron algorithm ------------------------------------------------------

train :: Weights -> TrainingSet -> Weights
train ws xs = let (ws', errorSum) = iterate' ws xs 0
              in if errorSum / n <= threshold
                  then ws'
                  else train ws' (traceShow (errorSum / n)  xs)
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
    where bin x | x + bias > 0 = 1
          bin x | otherwise = 0

adjustWeights :: Weights -> InputVector -> Error -> Weights
adjustWeights ws _ 0 = ws
adjustWeights ws xs e = [w + (learningRate * e * x) | (w, x) <- zip ws xs]


-- | To vector -----------------------------------------------------------------

-- | Convert an items of string types to a vector of float values.
--
-- TODO(jhibberd) List all fields in items.
-- TODO(jhibberd) Just don't include mapping for those that can't be 
-- interpreted.
-- TODO(jhibberd) Max 5 permutations, so formatter which converts actual value
-- to a category that appears in the permutation
toVector :: [String] -> [Float]
toVector xs = map (\(i, f) -> f (xs !! i)) converters
    where converters = (permMap 0 ["1", "2", "3"]) ++
                       [(1, quantifyName)] ++
                       (permMap 2 ["male", "female"]) ++ 
                       (permMap 3 ("":(map show [1..80]))) ++
                       [ (3, quantifyAge)
                       , (4, quantifySibsp)
                       , (5, quantifyParch)
                       , (6, quantifyTicket)
                       , (7, quantifyFare)
                       , (8, quantifyCabin)
                       , (9, quantifyEmbarked)
                       ]

permMap :: Int -> [String] -> [(Int, (String -> Float))]
permMap i m = map (\x -> (i, mapToFloat x)) $ permutations m

mapToFloat :: [String] -> String -> Float
mapToFloat xs x = case elemIndex x xs of
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

