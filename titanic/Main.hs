-- | Attempt to read from the test case CSV file.

{-
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard 
          Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances 
          Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.
-}

import Text.CSV
import Debug.Trace

filename = "/home/jhibberd/projects/learning/titanic/train.csv"
data Field 
    = Survived 
    | Pclass 
    | Name 
    | Sex 
    | Age 
    | Sibsp 
    | Parch 
    | Ticket 
    | Fare 
    | Cabin 
    | Embarked 
    deriving (Enum)

threshold = 0.27
bias = 0.5
learningRate = 0.001
initWeights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
-- [4.599864e-2,0.0,-2.0080369,1.8113816e-2,-0.5271171,-0.11799717,0.0,1.6099315e-2,0.0,2.9998511e-2]

main = do
    parseResult <- parseCSVFromFile filename
    let x = case parseResult of
                Left a -> error "Can't read file"
                Right a -> process $ stripHeadAndFoot a
    return $ train initWeights x

-- | Remove headers from empty line from CSV file.
stripHeadAndFoot :: [a] -> [a]
stripHeadAndFoot = tail . init

process :: [[String]] -> [([Float], Float)]
process xs = toTrainingSet $ map toVector xs
--where f x = x !! fromEnum Survived

toTrainingSet :: [[Float]] -> [([Float], Float)]
toTrainingSet = map (\x -> (tail x, head x))

type Weights = [Float]
type Sample = (InputVector, Float)
type TrainingSet = [Sample]
type InputVector = [Float]
type Error = Float
type ErrorSum = Float
type Classification = Float

train :: Weights -> TrainingSet -> Weights
train ws xs = let (ws', errorSum) = iterate' ws xs 0
                  errorSum' = traceShow errorSum errorSum
              in if errorSum' / n <= threshold
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
    where bin x | x + bias > 0 = 1
          bin x | otherwise = 0

adjustWeights :: Weights -> InputVector -> Error -> Weights
adjustWeights ws _ 0 = ws
adjustWeights ws xs e = [w + (learningRate * e * x) | (w, x) <- zip ws xs]


-- | To vector -----------------------------------------------------------------

toVector :: [String] -> [Float]
toVector xs = map (\(f, x) -> f x) $ zip converters xs
    where converters = 
            [ quantifySurvived
            , quantifyPclass
            , quantifyName
            , quantifySex
            , quantifyAge
            , quantifySibsp
            , quantifyParch
            , quantifyTicket
            , quantifyFare
            , quantifyCabin
            , quantifyEmbarked
            ]

quantifySurvived :: String -> Float
quantifySurvived = read

quantifyPclass :: String -> Float
quantifyPclass = read

-- | Virtually impossible to quantify name into a meaninful number.
quantifyName :: String -> Float
quantifyName x = 0

quantifySex :: String -> Float
quantifySex "male" = 1
quantifySex "female" = 0

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
quantifyFare = read

-- | Ignore for now. See 'quantifyTicket'.
quantifyCabin :: String -> Float
quantifyCabin x = 0

quantifyEmbarked :: String -> Float
quantifyEmbarked "" = 0
quantifyEmbarked "C" = 1
quantifyEmbarked "Q" = 2
quantifyEmbarked "S" = 3

