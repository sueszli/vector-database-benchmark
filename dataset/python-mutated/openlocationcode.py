import re
import math
SEPARATOR_ = '+'
SEPARATOR_POSITION_ = 8
PADDING_CHARACTER_ = '0'
CODE_ALPHABET_ = '23456789CFGHJMPQRVWX'
ENCODING_BASE_ = len(CODE_ALPHABET_)
LATITUDE_MAX_ = 90
LONGITUDE_MAX_ = 180
MAX_DIGIT_COUNT_ = 15
PAIR_CODE_LENGTH_ = 10
PAIR_FIRST_PLACE_VALUE_ = ENCODING_BASE_ ** (PAIR_CODE_LENGTH_ / 2 - 1)
PAIR_PRECISION_ = ENCODING_BASE_ ** 3
PAIR_RESOLUTIONS_ = [20.0, 1.0, 0.05, 0.0025, 0.000125]
GRID_CODE_LENGTH_ = MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_
GRID_COLUMNS_ = 4
GRID_ROWS_ = 5
GRID_LAT_FIRST_PLACE_VALUE_ = GRID_ROWS_ ** (GRID_CODE_LENGTH_ - 1)
GRID_LNG_FIRST_PLACE_VALUE_ = GRID_COLUMNS_ ** (GRID_CODE_LENGTH_ - 1)
FINAL_LAT_PRECISION_ = PAIR_PRECISION_ * GRID_ROWS_ ** (MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_)
FINAL_LNG_PRECISION_ = PAIR_PRECISION_ * GRID_COLUMNS_ ** (MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_)
MIN_TRIMMABLE_CODE_LEN_ = 6
GRID_SIZE_DEGREES_ = 0.000125

def isValid(code):
    if False:
        i = 10
        return i + 15
    '\n    Determines if a code is valid.\n    To be valid, all characters must be from the Open Location Code character\n    set with at most one separator. The separator can be in any even-numbered\n    position up to the eighth digit.\n    '
    sep = code.find(SEPARATOR_)
    if code.count(SEPARATOR_) > 1:
        return False
    if len(code) == 1:
        return False
    if sep == -1 or sep > SEPARATOR_POSITION_ or sep % 2 == 1:
        return False
    pad = code.find(PADDING_CHARACTER_)
    if pad != -1:
        if sep < SEPARATOR_POSITION_:
            return False
        if pad == 0:
            return False
        rpad = code.rfind(PADDING_CHARACTER_) + 1
        pads = code[pad:rpad]
        if len(pads) % 2 == 1 or pads.count(PADDING_CHARACTER_) != len(pads):
            return False
        if not code.endswith(SEPARATOR_):
            return False
    if len(code) - sep - 1 == 1:
        return False
    sepPad = SEPARATOR_ + PADDING_CHARACTER_
    for ch in code:
        if ch.upper() not in CODE_ALPHABET_ and ch not in sepPad:
            return False
    return True

def isShort(code):
    if False:
        print('Hello World!')
    '\n    Determines if a code is a valid short code.\n    A short Open Location Code is a sequence created by removing four or more\n    digits from an Open Location Code. It must include a separator\n    character.\n    '
    if not isValid(code):
        return False
    sep = code.find(SEPARATOR_)
    if sep >= 0 and sep < SEPARATOR_POSITION_:
        return True
    return False

def isFull(code):
    if False:
        print('Hello World!')
    '\n    Determines if a code is a valid full Open Location Code.\n    Not all possible combinations of Open Location Code characters decode to\n    valid latitude and longitude values. This checks that a code is valid\n    and also that the latitude and longitude values are legal. If the prefix\n    character is present, it must be the first character. If the separator\n    character is present, it must be after four characters.\n    '
    if not isValid(code):
        return False
    if isShort(code):
        return False
    firstLatValue = CODE_ALPHABET_.find(code[0].upper()) * ENCODING_BASE_
    if firstLatValue >= LATITUDE_MAX_ * 2:
        return False
    if len(code) > 1:
        firstLngValue = CODE_ALPHABET_.find(code[1].upper()) * ENCODING_BASE_
    if firstLngValue >= LONGITUDE_MAX_ * 2:
        return False
    return True

def encode(latitude, longitude, codeLength=PAIR_CODE_LENGTH_):
    if False:
        return 10
    '\n    Encode a location into an Open Location Code.\n    Produces a code of the specified length, or the default length if no length\n    is provided.\n    The length determines the accuracy of the code. The default length is\n    10 characters, returning a code of approximately 13.5x13.5 meters. Longer\n    codes represent smaller areas, but lengths > 14 are sub-centimetre and so\n    11 or 12 are probably the limit of useful codes.\n    Args:\n      latitude: A latitude in signed decimal degrees. Will be clipped to the\n          range -90 to 90.\n      longitude: A longitude in signed decimal degrees. Will be normalised to\n          the range -180 to 180.\n      codeLength: The number of significant digits in the output code, not\n          including any separator characters.\n    '
    if codeLength < 2 or (codeLength < PAIR_CODE_LENGTH_ and codeLength % 2 == 1):
        raise ValueError('Invalid Open Location Code length - ' + str(codeLength))
    codeLength = min(codeLength, MAX_DIGIT_COUNT_)
    latitude = clipLatitude(latitude)
    longitude = normalizeLongitude(longitude)
    if latitude == 90:
        latitude = latitude - computeLatitudePrecision(codeLength)
    code = ''
    latVal = int(round((latitude + LATITUDE_MAX_) * FINAL_LAT_PRECISION_, 6))
    lngVal = int(round((longitude + LONGITUDE_MAX_) * FINAL_LNG_PRECISION_, 6))
    if codeLength > PAIR_CODE_LENGTH_:
        for i in range(0, MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_):
            latDigit = latVal % GRID_ROWS_
            lngDigit = lngVal % GRID_COLUMNS_
            ndx = latDigit * GRID_COLUMNS_ + lngDigit
            code = CODE_ALPHABET_[ndx] + code
            latVal //= GRID_ROWS_
            lngVal //= GRID_COLUMNS_
    else:
        latVal //= pow(GRID_ROWS_, GRID_CODE_LENGTH_)
        lngVal //= pow(GRID_COLUMNS_, GRID_CODE_LENGTH_)
    for i in range(0, PAIR_CODE_LENGTH_ // 2):
        code = CODE_ALPHABET_[lngVal % ENCODING_BASE_] + code
        code = CODE_ALPHABET_[latVal % ENCODING_BASE_] + code
        latVal //= ENCODING_BASE_
        lngVal //= ENCODING_BASE_
    code = code[:SEPARATOR_POSITION_] + SEPARATOR_ + code[SEPARATOR_POSITION_:]
    if codeLength >= SEPARATOR_POSITION_:
        return code[0:codeLength + 1]
    return code[0:codeLength] + ''.zfill(SEPARATOR_POSITION_ - codeLength) + SEPARATOR_

def decode(code):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decodes an Open Location Code into the location coordinates.\n    Returns a CodeArea object that includes the coordinates of the bounding\n    box - the lower left, center and upper right.\n    Args:\n      code: The Open Location Code to decode.\n    Returns:\n      A CodeArea object that provides the latitude and longitude of two of the\n      corners of the area, the center, and the length of the original code.\n    '
    if not isFull(code):
        raise ValueError('Passed Open Location Code is not a valid full code - ' + str(code))
    code = re.sub('[+0]', '', code)
    code = code.upper()
    code = code[:MAX_DIGIT_COUNT_]
    normalLat = -LATITUDE_MAX_ * PAIR_PRECISION_
    normalLng = -LONGITUDE_MAX_ * PAIR_PRECISION_
    gridLat = 0
    gridLng = 0
    digits = min(len(code), PAIR_CODE_LENGTH_)
    pv = PAIR_FIRST_PLACE_VALUE_
    for i in range(0, digits, 2):
        normalLat += CODE_ALPHABET_.find(code[i]) * pv
        normalLng += CODE_ALPHABET_.find(code[i + 1]) * pv
        if i < digits - 2:
            pv //= ENCODING_BASE_
    latPrecision = float(pv) / PAIR_PRECISION_
    lngPrecision = float(pv) / PAIR_PRECISION_
    if len(code) > PAIR_CODE_LENGTH_:
        rowpv = GRID_LAT_FIRST_PLACE_VALUE_
        colpv = GRID_LNG_FIRST_PLACE_VALUE_
        digits = min(len(code), MAX_DIGIT_COUNT_)
        for i in range(PAIR_CODE_LENGTH_, digits):
            digitVal = CODE_ALPHABET_.find(code[i])
            row = digitVal // GRID_COLUMNS_
            col = digitVal % GRID_COLUMNS_
            gridLat += row * rowpv
            gridLng += col * colpv
            if i < digits - 1:
                rowpv //= GRID_ROWS_
                colpv //= GRID_COLUMNS_
        latPrecision = float(rowpv) / FINAL_LAT_PRECISION_
        lngPrecision = float(colpv) / FINAL_LNG_PRECISION_
    lat = float(normalLat) / PAIR_PRECISION_ + float(gridLat) / FINAL_LAT_PRECISION_
    lng = float(normalLng) / PAIR_PRECISION_ + float(gridLng) / FINAL_LNG_PRECISION_
    return CodeArea(round(lat, 14), round(lng, 14), round(lat + latPrecision, 14), round(lng + lngPrecision, 14), min(len(code), MAX_DIGIT_COUNT_))

def recoverNearest(code, referenceLatitude, referenceLongitude):
    if False:
        for i in range(10):
            print('nop')
    '\n     Recover the nearest matching code to a specified location.\n     Given a short code of between four and seven characters, this recovers\n     the nearest matching full code to the specified location.\n     Args:\n       code: A valid OLC character sequence.\n       referenceLatitude: The latitude (in signed decimal degrees) to use to\n           find the nearest matching full code.\n       referenceLongitude: The longitude (in signed decimal degrees) to use\n           to find the nearest matching full code.\n     Returns:\n       The nearest full Open Location Code to the reference location that matches\n       the short code. If the passed code was not a valid short code, but was a\n       valid full code, it is returned with proper capitalization but otherwise\n       unchanged.\n    '
    if isFull(code):
        return code.upper()
    if not isShort(code):
        raise ValueError('Passed short code is not valid - ' + str(code))
    referenceLatitude = clipLatitude(referenceLatitude)
    referenceLongitude = normalizeLongitude(referenceLongitude)
    code = code.upper()
    paddingLength = SEPARATOR_POSITION_ - code.find(SEPARATOR_)
    resolution = pow(20, 2 - paddingLength / 2)
    halfResolution = resolution / 2.0
    codeArea = decode(encode(referenceLatitude, referenceLongitude)[0:paddingLength] + code)
    if referenceLatitude + halfResolution < codeArea.latitudeCenter and codeArea.latitudeCenter - resolution >= -LATITUDE_MAX_:
        codeArea.latitudeCenter -= resolution
    elif referenceLatitude - halfResolution > codeArea.latitudeCenter and codeArea.latitudeCenter + resolution <= LATITUDE_MAX_:
        codeArea.latitudeCenter += resolution
    if referenceLongitude + halfResolution < codeArea.longitudeCenter:
        codeArea.longitudeCenter -= resolution
    elif referenceLongitude - halfResolution > codeArea.longitudeCenter:
        codeArea.longitudeCenter += resolution
    return encode(codeArea.latitudeCenter, codeArea.longitudeCenter, codeArea.codeLength)

def shorten(code, latitude, longitude):
    if False:
        for i in range(10):
            print('nop')
    '\n     Remove characters from the start of an OLC code.\n     This uses a reference location to determine how many initial characters\n     can be removed from the OLC code. The number of characters that can be\n     removed depends on the distance between the code center and the reference\n     location.\n     The minimum number of characters that will be removed is four. If more than\n     four characters can be removed, the additional characters will be replaced\n     with the padding character. At most eight characters will be removed.\n     The reference location must be within 50% of the maximum range. This ensures\n     that the shortened code will be able to be recovered using slightly different\n     locations.\n     Args:\n       code: A full, valid code to shorten.\n       latitude: A latitude, in signed decimal degrees, to use as the reference\n           point.\n       longitude: A longitude, in signed decimal degrees, to use as the reference\n           point.\n     Returns:\n       Either the original code, if the reference location was not close enough,\n       or the .\n    '
    if not isFull(code):
        raise ValueError('Passed code is not valid and full: ' + str(code))
    if code.find(PADDING_CHARACTER_) != -1:
        raise ValueError('Cannot shorten padded codes: ' + str(code))
    code = code.upper()
    codeArea = decode(code)
    if codeArea.codeLength < MIN_TRIMMABLE_CODE_LEN_:
        raise ValueError('Code length must be at least ' + MIN_TRIMMABLE_CODE_LEN_)
    latitude = clipLatitude(latitude)
    longitude = normalizeLongitude(longitude)
    coderange = max(abs(codeArea.latitudeCenter - latitude), abs(codeArea.longitudeCenter - longitude))
    for i in range(len(PAIR_RESOLUTIONS_) - 2, 0, -1):
        if coderange < PAIR_RESOLUTIONS_[i] * 0.3:
            return code[(i + 1) * 2:]
    return code

def clipLatitude(latitude):
    if False:
        print('Hello World!')
    '\n     Clip a latitude into the range -90 to 90.\n     Args:\n       latitude: A latitude in signed decimal degrees.\n    '
    return min(90, max(-90, latitude))

def computeLatitudePrecision(codeLength):
    if False:
        for i in range(10):
            print('nop')
    '\n     Compute the latitude precision value for a given code length. Lengths <=\n     10 have the same precision for latitude and longitude, but lengths > 10\n     have different precisions due to the grid method having fewer columns than\n     rows.\n    '
    if codeLength <= 10:
        return pow(20, math.floor(codeLength / -2 + 2))
    return pow(20, -3) / pow(GRID_ROWS_, codeLength - 10)

def normalizeLongitude(longitude):
    if False:
        i = 10
        return i + 15
    '\n     Normalize a longitude into the range -180 to 180, not including 180.\n     Args:\n       longitude: A longitude in signed decimal degrees.\n    '
    while longitude < -180:
        longitude = longitude + 360
    while longitude >= 180:
        longitude = longitude - 360
    return longitude

class CodeArea(object):
    """
     Coordinates of a decoded Open Location Code.
     The coordinates include the latitude and longitude of the lower left and
     upper right corners and the center of the bounding box for the area the
     code represents.
     Attributes:
       latitude_lo: The latitude of the SW corner in degrees.
       longitude_lo: The longitude of the SW corner in degrees.
       latitude_hi: The latitude of the NE corner in degrees.
       longitude_hi: The longitude of the NE corner in degrees.
       latitude_center: The latitude of the center in degrees.
       longitude_center: The longitude of the center in degrees.
       code_length: The number of significant characters that were in the code.
           This excludes the separator.
    """

    def __init__(self, latitudeLo, longitudeLo, latitudeHi, longitudeHi, codeLength):
        if False:
            return 10
        self.latitudeLo = latitudeLo
        self.longitudeLo = longitudeLo
        self.latitudeHi = latitudeHi
        self.longitudeHi = longitudeHi
        self.codeLength = codeLength
        self.latitudeCenter = min(latitudeLo + (latitudeHi - latitudeLo) / 2, LATITUDE_MAX_)
        self.longitudeCenter = min(longitudeLo + (longitudeHi - longitudeLo) / 2, LONGITUDE_MAX_)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return str([self.latitudeLo, self.longitudeLo, self.latitudeHi, self.longitudeHi, self.latitudeCenter, self.longitudeCenter, self.codeLength])

    def latlng(self):
        if False:
            while True:
                i = 10
        return [self.latitudeCenter, self.longitudeCenter]