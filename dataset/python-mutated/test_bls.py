import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time, TimeDelta
from astropy.timeseries.periodograms.bls import BoxLeastSquares
from astropy.timeseries.periodograms.lombscargle.core import has_units

def assert_allclose_blsresults(blsresult, other, **kwargs):
    if False:
        print('Hello World!')
    'Assert that another BoxLeastSquaresResults object is consistent\n\n    This method loops over all attributes and compares the values using\n    :func:`~astropy.tests.helper.assert_quantity_allclose` function.\n\n    Parameters\n    ----------\n    other : BoxLeastSquaresResults\n        The other results object to compare.\n\n    '
    for (k, v) in blsresult.items():
        if k not in other:
            raise AssertionError(f"missing key '{k}'")
        if k == 'objective':
            assert v == other[k], f"Mismatched objectives. Expected '{v}', got '{other[k]}'"
            continue
        assert_quantity_allclose(v, other[k], **kwargs)

@pytest.fixture
def data():
    if False:
        return 10
    t = np.array([6.96469186, 2.86139335, 2.26851454, 5.51314769, 7.1946897, 4.2310646, 9.80764198, 6.84829739, 4.80931901, 3.92117518, 3.43178016, 7.29049707, 4.38572245, 0.59677897, 3.98044255, 7.37995406, 1.8249173, 1.75451756, 5.31551374, 5.31827587, 6.34400959, 8.49431794, 7.24455325, 6.11023511, 7.22443383, 3.22958914, 3.61788656, 2.28263231, 2.93714046, 6.30976124, 0.9210494, 4.33701173, 4.30862763, 4.93685098, 4.2583029, 3.12261223, 4.26351307, 8.93389163, 9.44160018, 5.01836676, 6.23952952, 1.15618395, 3.17285482, 4.14826212, 8.66309158, 2.50455365, 4.83034264, 9.85559786, 5.19485119, 6.12894526, 1.20628666, 8.26340801, 6.03060128, 5.45068006, 3.42763834, 3.04120789, 4.17022211, 6.81300766, 8.75456842, 5.10422337, 6.69313783, 5.85936553, 6.24903502, 6.74689051, 8.42342438, 0.83194988, 7.63682841, 2.43666375, 1.94222961, 5.72456957, 0.95712517, 8.85326826, 6.27248972, 7.23416358, 0.16129207, 5.94431879, 5.56785192, 1.58959644, 1.53070515, 6.95529529, 3.18766426, 6.91970296, 5.5438325, 3.88950574, 9.2513249, 8.41669997, 3.57397567, 0.43591464, 3.04768073, 3.98185682, 7.0495883, 9.95358482, 3.55914866, 7.62547814, 5.93176917, 6.91701799, 1.51127452, 3.98876293, 2.40855898, 3.43456014, 5.13128154, 6.6662455, 1.05908485, 1.30894951, 3.21980606, 6.61564337, 8.46506225, 5.53257345, 8.54452488, 3.84837811, 3.16787897, 3.54264676, 1.71081829, 8.29112635, 3.38670846, 5.52370075, 5.78551468, 5.21533059, 0.02688065, 9.88345419, 9.05341576, 2.07635861, 2.92489413, 5.20010153, 9.01911373, 9.83630885, 2.57542064, 5.64359043, 8.06968684, 3.94370054, 7.31073036, 1.61069014, 6.00698568, 8.65864458, 9.83521609, 0.7936579, 4.28347275, 2.0454286, 4.50636491, 5.47763573, 0.9332671, 2.96860775, 9.2758424, 5.69003731, 4.57411998, 7.53525991, 7.41862152, 0.48579033, 7.08697395, 8.39243348, 1.65937884, 7.80997938, 2.86536617, 3.06469753, 6.65261465, 1.11392172, 6.64872449, 8.87856793, 6.96311268, 4.40327877, 4.38214384, 7.65096095, 5.65642001, 0.84904163, 5.82671088, 8.14843703, 3.37066383, 9.2757658, 7.50717, 5.74063825, 7.51643989, 0.79148961, 8.59389076, 8.21504113, 9.0987166, 1.28631198, 0.81780087, 1.38415573, 3.9937871, 4.24306861, 5.62218379, 1.2224355, 2.01399501, 8.11644348, 4.67987574, 8.07938209, 0.07426379, 5.51592726, 9.31932148, 5.82175459, 2.06095727, 7.17757562, 3.7898585, 6.68383947, 0.29319723, 6.35900359, 0.32197935, 7.44780655, 4.72913002, 1.21754355, 5.42635926, 0.66774443, 6.53364871, 9.96086327, 7.69397337, 5.73774114, 1.02635259, 6.99834075, 6.61167867, 0.49097131, 7.92299302, 5.18716591, 4.25867694, 7.88187174, 4.11569223, 4.81026276, 1.81628843, 3.213189, 8.45532997, 1.86903749, 4.17291061, 9.89034507, 2.36599812, 9.16832333, 9.18397468, 0.91296342, 4.63652725, 5.02216335, 3.1366895, 0.47339537, 2.41685637, 0.95529642, 2.38249906, 8.07791086, 8.94978288, 0.43222892, 3.01946836, 9.80582199, 5.39504823, 6.26309362, 0.05545408, 4.84909443, 9.88328535, 3.75185527, 0.97038159, 4.61908762, 9.63004466, 3.41830614, 7.98922733, 7.98846331, 2.08248297, 4.43367702, 7.15601275, 4.10519785, 1.91006955, 9.67494307, 6.50750366, 8.65459852, 0.252423578, 2.66905815, 5.020711, 0.674486351, 9.93033261, 2.36462396, 3.74292182, 2.14011915, 1.05445866, 2.32479786, 3.00610136, 6.34442268, 2.81234781, 3.62276761, 0.0594284372, 3.65719126, 5.33885982, 1.62015837, 5.97433108, 2.93152469, 6.32050495, 0.261966053, 8.8759346, 0.161186304, 1.26958031, 7.77162462, 0.458952322, 7.10998694, 9.71046141, 8.71682933, 7.10161651, 9.58509743, 4.29813338, 8.72878914, 3.55957668, 9.29763653, 1.48777656, 9.40029015, 8.32716197, 8.46054838, 1.2392301, 5.96486898, 0.163924809, 7.21184366, 0.0773751413, 0.848222774, 2.2549841, 8.75124534, 3.63576318, 5.39959935, 5.68103214, 2.2546336, 5.72146768, 6.60951795, 2.98245393, 4.18626859, 4.53088925, 9.32350662, 5.87493747, 9.48252372, 5.56034754, 5.00561421, 0.0353221097, 4.80889044, 9.27454999, 1.98365689, 0.520911344, 4.06778893, 3.72396481, 8.57153058, 0.266111156, 9.2014923, 6.80902999, 9.04225994, 6.07529071, 8.11953312, 3.35543874, 3.49566228, 3.8987423, 7.54797082, 3.69291174, 2.42219806, 9.37668357, 9.08011084, 3.48797316, 6.3463807, 2.73842212, 2.06115129, 3.36339529, 3.27099893, 8.82276101, 8.22303815, 7.09623229, 9.59345225, 4.22543353, 2.45033039, 1.17398437, 3.01053358, 1.45263734, 0.921860974, 6.02932197, 3.6418745, 5.64570343, 1.91335721, 6.7690586, 2.15505447, 2.78023594, 7.41760422, 5.59737896, 3.34836413, 5.42988783, 6.93984703, 9.12132121, 5.80713213, 2.32686379, 7.46697631, 7.77769018, 2.00401315, 8.2057422, 4.64934855, 7.79766662, 2.3747822, 3.3258027, 9.53697119, 6.57815073, 7.72877831, 6.88374343, 2.04304118, 4.70688748, 8.08963873, 6.75035127, 0.0602788565, 0.874077427, 3.4679472, 9.4436554, 4.91190481, 2.70176267, 3.60423719, 2.10652628, 4.21200057, 2.1803544, 8.45752507, 4.56270599, 2.79802018, 9.32891648, 3.14351354, 9.09714662, 0.43418091, 7.0711506, 4.83889039, 4.44221061, 0.363233444, 0.406831905, 3.32753617, 9.4711954, 6.17659977, 3.68874842, 6.11977039, 2.06131536, 1.65066443, 3.61817266, 8.63353352, 5.09401727, 2.96901516, 9.50251625, 8.1596609, 3.22973943, 9.72098245, 9.87351098, 4.08660134, 6.55923103, 4.05653198, 2.57348106, 0.82652676, 2.63610346, 2.71479854, 3.9863908, 1.84886031, 9.53818403, 1.02879885, 6.25208533, 4.41697388, 4.23518049, 3.71991783, 8.6831471, 2.80476981, 0.205761574, 9.18097016, 8.64480278, 2.7690179, 5.23487548, 1.09088197, 0.934270688, 8.37466108, 4.10265718, 6.6171654, 9.43200558, 2.45130592, 0.131598313, 0.241484058, 7.09385692, 9.24551885, 4.67330273, 3.75109148, 5.42860425, 8.58916838, 6.52153874, 2.32979897, 7.74580205, 1.34613497, 1.65559971, 6.12682283, 2.38783406, 7.04778548, 3.49518527, 2.7742396, 9.98918406, 0.406161246, 6.45822522, 0.38699585, 7.60210258, 2.30089957, 0.898318671, 6.48449712, 7.32601217, 6.78095315, 0.519009471, 2.94306946, 4.51088346, 2.8710329, 8.10513456, 1.31115105, 6.12179362, 9.88214944, 9.02556539, 2.22157062, 0.000818876137, 9.80597342, 8.82712985, 9.19472466, 4.15503551, 7.44615462])
    y = np.ones_like(t)
    dy = np.array([0.00606416, 0.00696152, 0.00925774, 0.00563806, 0.00946933, 0.00748254, 0.00713048, 0.00652823, 0.00958424, 0.00758812, 0.00902013, 0.00928826, 0.00961191, 0.0065169, 0.00669905, 0.00797537, 0.00720662, 0.00966421, 0.00698782, 0.00738889, 0.00808593, 0.0070237, 0.00996239, 0.00549426, 0.00610302, 0.00661328, 0.00573861, 0.0064211, 0.00889623, 0.00761446, 0.00516977, 0.00991311, 0.00808003, 0.0052947, 0.00830584, 0.00689185, 0.00567837, 0.00781832, 0.0086354, 0.00835563, 0.00623757, 0.00762433, 0.00768832, 0.00858402, 0.00679934, 0.00898866, 0.00813961, 0.00519166, 0.0077324, 0.00930956, 0.00783787, 0.00587914, 0.00755188, 0.00878473, 0.00555053, 0.0090855, 0.00583741, 0.00767038, 0.00692872, 0.00624312, 0.00823716, 0.00518696, 0.00880023, 0.0076347, 0.00937886, 0.00760359, 0.00517517, 0.005718, 0.00897802, 0.00745988, 0.0072094, 0.00659217, 0.00642275, 0.00982943, 0.00716485, 0.00942002, 0.00824082, 0.00929214, 0.00926225, 0.00978156, 0.00848971, 0.00902698, 0.00866564, 0.00802613, 0.00858677, 0.00857875, 0.00520454, 0.00758055, 0.00896326, 0.00621481, 0.00732574, 0.00717493, 0.00701394, 0.0056092, 0.00762856, 0.00723124, 0.00831696, 0.00774707, 0.00513771, 0.00515959, 0.0085068, 0.00853791, 0.0097997, 0.00938352, 0.0073403, 0.00812953, 0.00728591, 0.00611473, 0.00688338, 0.00551942, 0.00833264, 0.00596015, 0.00737734, 0.00983718, 0.00515834, 0.00575865, 0.0064929, 0.00970903, 0.00954421, 0.00581, 0.00990559, 0.00875374, 0.00769989, 0.00965851, 0.00940304, 0.00695658, 0.00828172, 0.00823693, 0.00663484, 0.00589695, 0.00733405, 0.00631641, 0.00677533, 0.00977072, 0.00730569, 0.00842446, 0.00668115, 0.00997931, 0.00829384, 0.00598005, 0.00549092, 0.0097159, 0.00972389, 0.00810664, 0.00508496, 0.00612767, 0.00900638, 0.0093773, 0.00726995, 0.0068276, 0.00637113, 0.00558485, 0.00557872, 0.00976301, 0.00904313, 0.0058239, 0.00603525, 0.00827776, 0.00882332, 0.00905157, 0.00581669, 0.00992064, 0.00613901, 0.00794708, 0.00793808, 0.00983681, 0.00828834, 0.00792452, 0.00759386, 0.00882329, 0.00553028, 0.00501046, 0.00976244, 0.00749329, 0.00664168, 0.00684027, 0.00901922, 0.00691185, 0.00885085, 0.00720231, 0.00922039, 0.00538102, 0.00740564, 0.00733425, 0.00632164, 0.00971807, 0.00952514, 0.00721798, 0.0054858, 0.00603392, 0.00635746, 0.0074211, 0.00669189, 0.00887068, 0.00738013, 0.00935185, 0.00997891, 0.00609918, 0.00805836, 0.00923751, 0.00972618, 0.00645043, 0.00863521, 0.00507508, 0.00939571, 0.00531969, 0.00866698, 0.00997305, 0.00750595, 0.00604667, 0.00797322, 0.00812075, 0.00834036, 0.00586306, 0.00949356, 0.00810496, 0.00521784, 0.00842021, 0.00598042, 0.0051367, 0.00775477, 0.00906657, 0.00929971, 0.0055176, 0.00831521, 0.00855038, 0.00647258, 0.00985682, 0.00639344, 0.00534991, 0.0075964, 0.00847157, 0.0062233, 0.00669291, 0.00781814, 0.00943339, 0.00873663, 0.00604796, 0.00625889, 0.0076194, 0.00884479, 0.00809381, 0.00750662, 0.00798563, 0.0087803, 0.0076854, 0.00948876, 0.00973534, 0.00957677, 0.00877259, 0.00623161, 0.00692636, 0.0064, 0.0082883, 0.00662111, 0.00877196, 0.00556755, 0.00887682, 0.00792951, 0.00917694, 0.00715438, 0.00812482, 0.00777206, 0.00987836, 0.00877737, 0.00772407, 0.00587016, 0.00952057, 0.00602919, 0.00825022, 0.00968236, 0.0061179, 0.00612962, 0.00925909, 0.00913828, 0.00675852, 0.00632548, 0.00563694, 0.00993968, 0.00917672, 0.00949696, 0.0075684, 0.00557192, 0.0052629, 0.00665291, 0.00960165, 0.00973791, 0.00920582, 0.0057934, 0.00709962, 0.00623121, 0.00602675, 0.00842413, 0.00743056, 0.00662455, 0.00550107, 0.00772382, 0.00673513, 0.00695548, 0.00655254, 0.00693598, 0.0077793, 0.00507072, 0.00923823, 0.0096096, 0.00775265, 0.00634011, 0.0099512, 0.00691597, 0.00846828, 0.00844976, 0.00717155, 0.00599579, 0.0098329, 0.00531845, 0.00742575, 0.00610365, 0.00646987, 0.00914264, 0.00683633, 0.00541674, 0.00598155, 0.00930187, 0.00988514, 0.00633991, 0.00837704, 0.00540599, 0.00861733, 0.00708218, 0.0095908, 0.00655768, 0.00970733, 0.00751624, 0.00674446, 0.0082351, 0.00624873, 0.00614882, 0.00598173, 0.0097995, 0.00746457, 0.00875807, 0.00736996, 0.0079377, 0.00792069, 0.00989943, 0.00834217, 0.00619885, 0.00507599, 0.00609341, 0.0072776, 0.0069671, 0.00906163, 0.00892778, 0.00544548, 0.00976005, 0.00763728, 0.00798202, 0.00702528, 0.0082475, 0.00935663, 0.00836968, 0.00985049, 0.00850561, 0.0091086, 0.0052252, 0.00836349, 0.00827376, 0.00550873, 0.00921194, 0.00807086, 0.00549164, 0.00797234, 0.00739208, 0.00616647, 0.00509878, 0.00682784, 0.00809926, 0.0066464, 0.00653627, 0.00875561, 0.00879312, 0.00859383, 0.00550591, 0.00758083, 0.00778899, 0.00872402, 0.00951589, 0.00684519, 0.00714332, 0.00866384, 0.00831318, 0.00778935, 0.0067507, 0.00597676, 0.00591904, 0.00540792, 0.005406, 0.00922899, 0.00691836, 0.0053037, 0.00948213, 0.00611635, 0.00634062, 0.00597249, 0.00983751, 0.0055627, 0.00861082, 0.00966044, 0.00834001, 0.00929363, 0.00621224, 0.00836964, 0.00850436, 0.00729166, 0.00935273, 0.00847193, 0.00947439, 0.00876602, 0.00760145, 0.00749344, 0.00726864, 0.00510823, 0.00767571, 0.00711487, 0.00578767, 0.00559535, 0.00724676, 0.00519957, 0.0099329, 0.0068906, 0.00691055, 0.00525563, 0.00713336, 0.00507873, 0.00515047, 0.0066955, 0.00910484, 0.00729411, 0.0050742, 0.0058161, 0.00869961, 0.00869147, 0.00877261, 0.00675835, 0.00676138, 0.00901038, 0.00699069, 0.00863596, 0.00790562, 0.00682171, 0.00540003, 0.00558063, 0.00944779, 0.0072617, 0.00997002, 0.00681948, 0.00624977, 0.0067527, 0.00671543, 0.00818678, 0.00506369, 0.00881634, 0.00708207, 0.0071612, 0.00740558, 0.00724606, 0.00748735, 0.00672952, 0.00726673, 0.00702326, 0.00759121, 0.00811635, 0.0062052, 0.00754219, 0.00797311, 0.00508474, 0.00760247, 0.00619647, 0.00702269, 0.00913265, 0.00663118, 0.00741608, 0.00512371, 0.00654375, 0.00819861, 0.00657581, 0.00602899, 0.00645328, 0.00977189, 0.00543401, 0.00731679, 0.00529193, 0.00769329, 0.00573018, 0.00817042, 0.00632199, 0.00845458, 0.00673573, 0.00502084, 0.00647447])
    period = 2.0
    transit_time = 0.5
    duration = 0.16
    depth = 0.2
    m = np.abs((t - transit_time + 0.5 * period) % period - 0.5 * period) < 0.5 * duration
    y[m] = 1.0 - depth
    randn_arr = np.array([-0.0100326528, -0.845644428, 0.91146061, -1.37449688, -0.547065645, -7.55266106e-05, -0.121166803, -2.00858547, -0.920646543, 0.168234342, -1.31989156, 1.2664293, 0.495180889, -0.514240391, -0.220292465, 1.86156412, 0.935988451, 0.380219145, -1.41551877, 1.62961132, 1.05240107, -0.148405388, -0.549698069, -0.187903939, -1.20193668, -0.470785558, 0.763160514, -1.80762128, -0.314074374, 0.113755973, 0.103568037, -1.17893695, -1.18215289, 1.08916538, -1.22452909, 1.00865096, -0.482365315, 1.07979635, -0.421078505, -1.16647132, 0.856554856, -0.0173912222, 1.44857659, 0.892200085, -0.229426629, -0.449667602, 0.0233723433, 0.190210018, -0.881748527, 0.841939573, -0.397363492, -0.423027745, -0.540688337, 0.231017267, -0.692052602, 0.13497011, 2.76660307, -0.0536094601, -0.434004738, -1.66768923, 0.0502219248, -1.10923094, -0.375558119, 0.151607594, -1.73098945, 0.157462752, 0.304515175, -1.29710002, -0.392309192, -1.83066636, 1.57550094, 0.330563277, -0.179588501, -0.163435831, 1.13144361, -0.0941655519, 0.330816771, 1.51862956, -0.346167148, -1.09263532, -0.824500575, 1.42866383, 0.0914283085, -0.502331288, 0.97364438, 0.997957386, -0.475647768, -0.971936837, -1.5705286, -1.79388892, -0.264986452, -0.893195947, 1.85847441, 0.0585377547, -1.94214954, 1.41872928, 0.161710309, 0.70497948, 0.682034777, 0.296556567, 0.52334263, 0.238760672, -1.10638591, 0.366732198, 1.0239055, -0.210056413, 0.551302218, 0.419589145, 1.81565206, -0.252750301, -0.292004163, -0.11693174, -0.102391075, -2.27261771, -0.642609841, 0.299885067, -0.00825651467, -0.799339154, -0.664779252, -0.355613128, -0.801571781, -0.51305061, -0.539390119, 0.895370847, 1.01639127, 0.933585094, 0.426701799, -0.708322484, 0.95983045, -0.314250587, 0.0230522083, 1.33822053, 0.0839928561, 0.24728403, -1.41277949, 0.487009294, -0.980006647, 1.01193966, -0.184599177, -2.23616884, -0.358020103, -0.228034538, 0.485475226, 0.670512391, -0.327764245, 1.01286819, -3.16705533, -0.713988998, -1.11236427, -1.25418351, 0.959706371, 0.829170399, -0.77577002, 1.178057, 0.101466892, -0.421684101, -0.692922796, -0.778271726, 0.472774857, 0.650154901, 0.238501212, -2.05021768, 0.296358656, 0.565396564, -0.669205605, 0.0432505429, -1.8638843, -1.22996906, -0.324235348, -0.309751144, 0.351679372, -1.18692539, -0.341206065, -0.48977978, 0.528010474, 1.42104277, 1.72092032, -1.56844005, -0.0480141918, -1.11252931, -0.0647449515, 0.42291928, 0.0814908987, -0.0490116988, 1.48303917, 0.720989392, -0.272654462, 0.0242113609, 0.870897807, 0.609790506, -0.425076104, -1.77524284, -1.18465749, 0.145979225, -1.78652685, -0.152394498, -0.453569176, 0.999252803, -1.31804382, -1.93176898, -0.419640742, 0.634763132, 1.0699186, -0.909327017, 0.470263748, -1.11143045, -0.748827466, 0.567594726, 0.718150543, -0.999380749, 0.474898323, -1.86849981, -0.202658907, -1.13424803, -0.80769934, -1.27607735, 0.553626395, 0.55387447, -0.691200445, 0.375582306, 0.261272553, -0.128451754, 2.1581702, -0.840878617, 0.0143050907, -0.382387029, -0.371780015, 0.159412004, -0.2943957, -0.86042676, 0.124227498, 1.18233165, 0.94276638, 0.203044488, -0.735396814, 0.1864296, 1.08464302, 1.19118926, 0.35968706, -0.3643572, -0.202752749, 0.772045927, 0.686346215, -1.75769961, 0.658617565, 0.71128834, -0.887191425, -0.764981116, -0.757164098, -0.680262803, -1.41674959, 0.31309193, -0.785719399, -0.0703838361, -0.497568783, 0.255177521, -1.01061704, 0.245265375, 0.389781016, 0.827594585, 1.96776909, -2.09210177, 0.320314334, -0.709162842, -1.92505867, 0.841630623, 1.33219988, -0.39162771, 0.210916296, -0.0640767402, 0.434197668, 0.880535749, 0.344937336, 0.345769929, 1.25973654, -0.164662222, 0.923064571, -0.822000422, 1.60708495, 0.737825392, -0.403759534, -2.11454815, -0.000310717131, -1.18180941, 0.299634603, 1.45116882, 0.160059793, -0.178012614, 0.342205404, 0.285650196, -2.36286411, 0.240936864, 0.620277356, -0.259341634, 0.978559078, -0.127674575, 0.766998762, 2.27310511, -0.096391129, -1.94213217, -0.336591724, -1.72589, 0.611237826, 1.30935097, 0.695879662, 0.320308213, -0.644925458, 1.57564975, 0.753276212, 0.284469557, 0.204860319, 0.111627359, 0.452216424, -0.613327179, 1.52524993, 0.152339753, 0.60005445, -0.433567278, 0.374918534, -2.28175243, -1.11829888, -0.0314131532, -1.32247311, 2.43941406, -1.66808131, 0.345900749, 1.65577315, 0.481287059, -0.310227553, -0.552144084, 0.673255489, -0.800270681, -0.11948611, 0.691198606, -0.307879027, 0.0875100102, -0.304086293, -0.969797604, 1.18915048, 1.39306624, -0.316699954, -0.265576159, -0.177899339, 0.538803274, -0.905300265, -0.0885253056, 0.262959055, 0.642042149, -2.78083727, 0.40340321, 0.345846762, 1.00772824, -0.526264015, -0.518353205, 1.20251659, -1.56315671, 1.62909029, 2.55589446, 0.477451685, 0.814098474, -1.48958171, -0.694559787, 1.05786255, 0.361815347, -0.181427463, 0.232869132, 0.506976484, -0.293095701, -0.028945945, -0.0363073748, -1.05227898, 0.323594628, 1.80358591, 1.73196213, -1.4763993, 0.57063122, 0.675503781, -0.410510463, -0.964200035, -1.32081431, -0.444703779, 0.350009137, -0.158058176, -0.610933088, -1.24915663, 0.350716258, 1.06654245, -0.926921972, 0.448428964, -1.87947524, -0.657466109, 0.72960412, -1.11776721, -0.604436725, 1.41796683, -0.73284398, -0.853944819, 0.575848362, 1.95473356, -0.239669947, 0.76873586, 1.34576918, 0.325552163, -0.269917901, -0.876326739, -1.42521096, 1.11170175, 0.180957146, 1.33280094, 0.988925316, -0.61697052, -1.1868867, 0.412669583, -0.632506884, 0.376689141, -0.731151938, -0.861225253, -0.14099081, 0.93410062, 0.306539895, 1.17837515, -1.2335617, -1.05707714, -0.0891636992, 2.16570138, 0.674286114, -1.06661274, -0.076140453, 0.220714791, -0.568685746, 0.613274991, -0.156446138, -0.299330718, 1.26025679, -1.7096609, -0.961805342, -0.817308981, -0.84768107, -0.728753045, 0.488475958, 1.09653283, 0.916041261, -1.01956213, -0.107417899, 0.452265213, 0.240002952, 1.3057474, -0.675334236, 0.156319421, -0.393230715, 0.251075019, -1.07889691, -0.928937721, -0.73011086, -0.563669311, 1.54792327, 1.17540191, -0.212649671, 0.172933294, -1.59443602, -0.179292347, 0.159614713, 1.14568421, 0.32680472, 0.432890059, 0.29776289, 0.26900119, -1.39675918, -0.416757668, 1.4348868, 0.823896443, 0.494234499, 0.0667153092, 0.659441396, -0.944889409, -1.58005956, -0.382086552, 0.537923058, 0.107829882, 1.01395868, 0.351450517, 0.0448421962, 1.32748495, 1.13237578, -0.0980913012, -1.10304986, -0.907361492, -0.161451138, -0.366811384, 1.65776233, -1.68013415, -0.0642577869, -1.06622649, 0.116801869, 0.382264833, -0.404896974, 0.530481414, -0.198626941, -0.179395613, -0.417888725])
    y += dy * randn_arr
    return (t, y, dy, dict(period=period, transit_time=transit_time, duration=duration, depth=depth))

def test_32bit_bug():
    if False:
        for i in range(10):
            print('nop')
    rand = np.random.default_rng(42)
    t = rand.uniform(0, 10, 500)
    y = np.ones_like(t)
    y[np.abs((t + 1.0) % 2.0 - 1) < 0.08] = 1.0 - 0.1
    y += 0.01 * rand.standard_normal(len(t))
    model = BoxLeastSquares(t, y)
    results = model.autopower(0.16)
    assert_allclose(results.period[np.argmax(results.power)], 2.000412388152837)
    periods = np.linspace(1.9, 2.1, 5)
    results = model.power(periods, 0.16)
    assert_allclose(results.power, [0.01723948, 0.0643028, 0.1338783, 0.09428816, 0.03577543], rtol=1.1e-07)

@pytest.mark.parametrize('objective', ['likelihood', 'snr'])
def test_correct_model(data, objective):
    if False:
        while True:
            i = 10
    (t, y, dy, params) = data
    model = BoxLeastSquares(t, y, dy)
    periods = np.exp(np.linspace(np.log(params['period']) - 0.1, np.log(params['period']) + 0.1, 1000))
    results = model.power(periods, params['duration'], objective=objective)
    ind = np.argmax(results.power)
    for (k, v) in params.items():
        assert_allclose(results[k][ind], v, atol=0.01)
    chi = (results.depth[ind] - params['depth']) / results.depth_err[ind]
    assert np.abs(chi) < 1

@pytest.mark.parametrize('objective', ['likelihood', 'snr'])
@pytest.mark.parametrize('offset', [False, True])
def test_fast_method(data, objective, offset):
    if False:
        print('Hello World!')
    (t, y, dy, params) = data
    if offset:
        t = t - params['transit_time'] + params['period']
    model = BoxLeastSquares(t, y, dy)
    periods = np.exp(np.linspace(np.log(params['period']) - 1, np.log(params['period']) + 1, 10))
    durations = params['duration']
    results = model.power(periods, durations, objective=objective)
    assert_allclose_blsresults(results, model.power(periods, durations, method='slow', objective=objective))

def test_input_units(data):
    if False:
        for i in range(10):
            print('nop')
    (t, y, dy, params) = data
    t_unit = u.day
    y_unit = u.mag
    with pytest.raises(u.UnitConversionError):
        BoxLeastSquares(t * t_unit, y * y_unit, dy * u.one)
    with pytest.raises(u.UnitConversionError):
        BoxLeastSquares(t * t_unit, y * u.one, dy * y_unit)
    with pytest.raises(u.UnitConversionError):
        BoxLeastSquares(t * t_unit, y, dy * y_unit)
    model = BoxLeastSquares(t * t_unit, y * u.one, dy)
    assert model.dy.unit == model.y.unit
    model = BoxLeastSquares(t * t_unit, y * y_unit, dy)
    assert model.dy.unit == model.y.unit
    model = BoxLeastSquares(t * t_unit, y * y_unit)
    assert model.dy is None

def test_period_units(data):
    if False:
        for i in range(10):
            print('nop')
    (t, y, dy, params) = data
    t_unit = u.day
    y_unit = u.mag
    model = BoxLeastSquares(t * t_unit, y * y_unit, dy)
    p = model.autoperiod(params['duration'])
    assert p.unit == t_unit
    p = model.autoperiod(params['duration'] * 24 * u.hour)
    assert p.unit == t_unit
    with pytest.raises(u.UnitConversionError):
        model.autoperiod(params['duration'] * u.mag)
    p = model.autoperiod(params['duration'], minimum_period=0.5)
    assert p.unit == t_unit
    with pytest.raises(u.UnitConversionError):
        p = model.autoperiod(params['duration'], minimum_period=0.5 * u.mag)
    p = model.autoperiod(params['duration'], maximum_period=0.5)
    assert p.unit == t_unit
    with pytest.raises(u.UnitConversionError):
        p = model.autoperiod(params['duration'], maximum_period=0.5 * u.mag)
    p = model.autoperiod(params['duration'], minimum_period=0.5, maximum_period=1.5)
    p2 = model.autoperiod(params['duration'], maximum_period=0.5, minimum_period=1.5)
    assert_quantity_allclose(p, p2)

@pytest.mark.parametrize('method', ['fast', 'slow'])
@pytest.mark.parametrize('with_err', [True, False])
@pytest.mark.parametrize('t_unit', [None, u.day])
@pytest.mark.parametrize('y_unit', [None, u.mag])
@pytest.mark.parametrize('objective', ['likelihood', 'snr'])
def test_results_units(data, method, with_err, t_unit, y_unit, objective):
    if False:
        for i in range(10):
            print('nop')
    (t, y, dy, params) = data
    periods = np.linspace(params['period'] - 1.0, params['period'] + 1.0, 3)
    if t_unit is not None:
        t = t * t_unit
    if y_unit is not None:
        y = y * y_unit
        dy = dy * y_unit
    if not with_err:
        dy = None
    model = BoxLeastSquares(t, y, dy)
    results = model.power(periods, params['duration'], method=method, objective=objective)
    if t_unit is None:
        assert not has_units(results.period)
        assert not has_units(results.duration)
        assert not has_units(results.transit_time)
    else:
        assert results.period.unit == t_unit
        assert results.duration.unit == t_unit
        assert results.transit_time.unit == t_unit
    if y_unit is None:
        assert not has_units(results.power)
        assert not has_units(results.depth)
        assert not has_units(results.depth_err)
        assert not has_units(results.depth_snr)
        assert not has_units(results.log_likelihood)
    else:
        assert results.depth.unit == y_unit
        assert results.depth_err.unit == y_unit
        assert results.depth_snr.unit == u.one
        if dy is None:
            assert results.log_likelihood.unit == y_unit * y_unit
            if objective == 'snr':
                assert results.power.unit == u.one
            else:
                assert results.power.unit == y_unit * y_unit
        else:
            assert results.log_likelihood.unit == u.one
            assert results.power.unit == u.one

def test_autopower(data):
    if False:
        print('Hello World!')
    (t, y, dy, params) = data
    duration = params['duration'] + np.linspace(-0.1, 0.1, 3)
    model = BoxLeastSquares(t, y, dy)
    period = model.autoperiod(duration)
    results1 = model.power(period, duration)
    results2 = model.autopower(duration)
    assert_allclose_blsresults(results1, results2)

@pytest.mark.parametrize('with_units', [True, False])
def test_model(data, with_units):
    if False:
        for i in range(10):
            print('nop')
    (t, y, dy, params) = data
    A = np.zeros((len(t), 2))
    p = params['period']
    dt = np.abs((t - params['transit_time'] + 0.5 * p) % p - 0.5 * p)
    m_in = dt < 0.5 * params['duration']
    A[~m_in, 0] = 1.0
    A[m_in, 1] = 1.0
    w = np.linalg.solve(np.dot(A.T, A / dy[:, None] ** 2), np.dot(A.T, y / dy ** 2))
    model_true = np.dot(A, w)
    if with_units:
        t = t * u.day
        y = y * u.mag
        dy = dy * u.mag
        model_true = model_true * u.mag
    pgram = BoxLeastSquares(t, y, dy)
    model = pgram.model(t, p, params['duration'], params['transit_time'])
    transit_mask = pgram.transit_mask(t, p, params['duration'], params['transit_time'])
    transit_mask0 = model - model.max() < 0.0
    assert_allclose(transit_mask, transit_mask0)
    assert_quantity_allclose(model, model_true)

@pytest.mark.parametrize('shape', [(1,), (2,), (3,), (2, 3)])
def test_shapes(data, shape):
    if False:
        print('Hello World!')
    (t, y, dy, params) = data
    duration = params['duration']
    model = BoxLeastSquares(t, y, dy)
    period = np.empty(shape)
    period.flat = np.linspace(params['period'] - 1, params['period'] + 1, period.size)
    if len(period.shape) > 1:
        with pytest.raises(ValueError):
            results = model.power(period, duration)
    else:
        results = model.power(period, duration)
        for (k, v) in results.items():
            if k == 'objective':
                continue
            assert v.shape == shape

@pytest.mark.parametrize('with_units', [True, False])
@pytest.mark.parametrize('with_err', [True, False])
def test_compute_stats(data, with_units, with_err):
    if False:
        i = 10
        return i + 15
    (t, y, dy, params) = data
    y_unit = 1
    if with_units:
        y_unit = u.mag
        t = t * u.day
        y = y * u.mag
        dy = dy * u.mag
        params['period'] = params['period'] * u.day
        params['duration'] = params['duration'] * u.day
        params['transit_time'] = params['transit_time'] * u.day
        params['depth'] = params['depth'] * u.mag
    if not with_err:
        dy = None
    model = BoxLeastSquares(t, y, dy)
    results = model.power(params['period'], params['duration'], oversample=1000)
    stats = model.compute_stats(params['period'], params['duration'], params['transit_time'])
    tt = params['period'] * np.arange(int(t.max() / params['period']) + 1)
    tt += params['transit_time']
    assert_quantity_allclose(tt, stats['transit_times'])
    assert_allclose(stats['per_transit_count'], [9, 7, 7, 7, 8])
    assert_quantity_allclose(np.sum(stats['per_transit_log_likelihood']), results['log_likelihood'])
    assert_quantity_allclose(stats['depth'][0], results['depth'])
    results_half = model.power(0.5 * params['period'], params['duration'], oversample=1000)
    assert_quantity_allclose(stats['depth_half'][0], results_half['depth'])
    if not with_err:
        assert_quantity_allclose(stats['harmonic_amplitude'], 0.029945029964964204 * y_unit)
        assert_quantity_allclose(stats['harmonic_delta_log_likelihood'], -0.5875918155223113 * y_unit * y_unit)
        return
    assert_quantity_allclose(stats['harmonic_amplitude'], 0.03302798874227585 * y_unit)
    assert_quantity_allclose(stats['harmonic_delta_log_likelihood'], -12407.505922833765)
    assert_quantity_allclose(stats['depth'][1], results['depth_err'])
    assert_quantity_allclose(stats['depth_half'][1], results_half['depth_err'])
    for (f, k) in zip((1.0, 1.0, 1.0, 0.0), ('depth', 'depth_even', 'depth_odd', 'depth_phased')):
        res = np.abs((stats[k][0] - f * params['depth']) / stats[k][1])
        assert res < 1, f'f={f}, k={k}, res={res}'

def test_negative_times(data):
    if False:
        print('Hello World!')
    (t, y, dy, params) = data
    mu = np.mean(t)
    duration = params['duration'] + np.linspace(-0.1, 0.1, 3)
    model1 = BoxLeastSquares(t, y, dy)
    results1 = model1.autopower(duration)
    model2 = BoxLeastSquares(t - mu, y, dy)
    results2 = model2.autopower(duration)
    results2.transit_time = (results2.transit_time + mu) % results2.period
    assert_allclose_blsresults(results1, results2)

@pytest.mark.parametrize('timedelta', [False, True])
def test_absolute_times(data, timedelta):
    if False:
        i = 10
        return i + 15
    (t, y, dy, params) = data
    t = t * u.day
    y = y * u.mag
    dy = dy * u.mag
    start = Time('2019-05-04T12:34:56')
    trel = TimeDelta(t) if timedelta else t
    t = trel + start
    bls1 = BoxLeastSquares(t, y, dy)
    bls2 = BoxLeastSquares(trel, y, dy)
    results1 = bls1.autopower(0.16 * u.day)
    results2 = bls2.autopower(0.16 * u.day)
    for key in results1:
        if key == 'transit_time':
            assert_quantity_allclose((results1[key] - start).to(u.day), results2[key])
        elif key == 'objective':
            assert results1[key] == results2[key]
        else:
            assert_allclose(results1[key], results2[key])
    model1 = bls1.model(t, 0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    model2 = bls2.model(trel, 0.2 * u.day, 0.05 * u.day, TimeDelta(1 * u.day))
    assert_quantity_allclose(model1, model2)
    MESSAGE = '{} was provided as {} time but the BoxLeastSquares class was initialized with {} times\\.'
    with pytest.raises(TypeError, match=MESSAGE.format('transit_time', 'a relative', 'absolute')):
        bls1.model(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)
    with pytest.raises(TypeError, match=MESSAGE.format('t_model', 'a relative', 'absolute')):
        bls1.model(trel, 0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    with pytest.raises(TypeError, match=MESSAGE.format('transit_time', 'an absolute', 'relative')):
        bls2.model(trel, 0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    with pytest.raises(TypeError, match=MESSAGE.format('t_model', 'an absolute', 'relative')):
        bls2.model(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)
    stats1 = bls1.compute_stats(0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    stats2 = bls2.compute_stats(0.2 * u.day, 0.05 * u.day, 1 * u.day)
    for key in stats1:
        if key == 'transit_times':
            assert_quantity_allclose((stats1[key] - start).to(u.day), stats2[key], atol=1e-10 * u.day)
        elif key.startswith('depth'):
            for (value1, value2) in zip(stats1[key], stats2[key]):
                assert_quantity_allclose(value1, value2)
        else:
            assert_allclose(stats1[key], stats2[key])
    MESSAGE = '{} was provided as {} time but the BoxLeastSquares class was initialized with {} times\\.'
    with pytest.raises(TypeError, match=MESSAGE.format('transit_time', 'a relative', 'absolute')):
        bls1.compute_stats(0.2 * u.day, 0.05 * u.day, 1 * u.day)
    with pytest.raises(TypeError, match=MESSAGE.format('transit_time', 'an absolute', 'relative')):
        bls2.compute_stats(0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    mask1 = bls1.transit_mask(t, 0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    mask2 = bls2.transit_mask(trel, 0.2 * u.day, 0.05 * u.day, 1 * u.day)
    assert_equal(mask1, mask2)
    with pytest.raises(TypeError, match=MESSAGE.format('transit_time', 'a relative', 'absolute')):
        bls1.transit_mask(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)
    with pytest.raises(TypeError, match=MESSAGE.format('t', 'a relative', 'absolute')):
        bls1.transit_mask(trel, 0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    with pytest.raises(TypeError, match=MESSAGE.format('transit_time', 'an absolute', 'relative')):
        bls2.transit_mask(trel, 0.2 * u.day, 0.05 * u.day, Time('2019-06-04T12:34:56'))
    with pytest.raises(TypeError, match=MESSAGE.format('t', 'an absolute', 'relative')):
        bls2.transit_mask(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)

def test_transit_time_in_range(data):
    if False:
        print('Hello World!')
    (t, y, dy, params) = data
    t_ref = 10230.0
    t2 = t + t_ref
    bls1 = BoxLeastSquares(t, y, dy)
    bls2 = BoxLeastSquares(t2, y, dy)
    results1 = bls1.autopower(0.16)
    results2 = bls2.autopower(0.16)
    assert np.allclose(results1.transit_time, results2.transit_time - t_ref)
    assert np.all(results1.transit_time >= t.min())
    assert np.all(results1.transit_time <= t.max())
    assert np.all(results2.transit_time >= t2.min())
    assert np.all(results2.transit_time <= t2.max())