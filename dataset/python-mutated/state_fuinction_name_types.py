from enum import Enum
from localstack.services.stepfunctions.asl.antlr.runtime.ASLIntrinsicLexer import ASLIntrinsicLexer

class StatesFunctionNameType(Enum):
    Format = ASLIntrinsicLexer.Format
    StringToJson = ASLIntrinsicLexer.StringToJson
    JsonToString = ASLIntrinsicLexer.JsonToString
    Array = ASLIntrinsicLexer.Array
    ArrayPartition = ASLIntrinsicLexer.ArrayPartition
    ArrayContains = ASLIntrinsicLexer.ArrayContains
    ArrayRange = ASLIntrinsicLexer.ArrayRange
    ArrayGetItem = ASLIntrinsicLexer.ArrayGetItem
    ArrayLength = ASLIntrinsicLexer.ArrayLength
    ArrayUnique = ASLIntrinsicLexer.ArrayUnique
    Base64Encode = ASLIntrinsicLexer.Base64Encode
    Base64Decode = ASLIntrinsicLexer.Base64Decode
    Hash = ASLIntrinsicLexer.Hash
    JsonMerge = ASLIntrinsicLexer.JsonMerge
    MathRandom = ASLIntrinsicLexer.MathRandom
    MathAdd = ASLIntrinsicLexer.MathAdd
    StringSplit = ASLIntrinsicLexer.StringSplit
    UUID = ASLIntrinsicLexer.UUID

    def name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ASLIntrinsicLexer.literalNames[self.value][1:-1]