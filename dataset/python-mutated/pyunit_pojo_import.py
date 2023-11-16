import h2o
from h2o.estimators import H2OGradientBoostingEstimator, H2OGeneralizedLinearEstimator
from tests import pyunit_utils
import os
import sys
from pandas.testing import assert_frame_equal
TEMPLATE = '\nimport java.util.HashMap;\nimport java.util.Map;\nimport hex.genmodel.GenModel;\nimport hex.genmodel.annotations.ModelPojo;\n\npublic class %s extends GenModel {\n\n    public hex.ModelCategory getModelCategory() { return hex.ModelCategory.Regression; }\n    public boolean isSupervised() { return true; }\n    public int nfeatures() { return 19; }\n    public int nclasses() { return 1; } // use "1" for regression\n\n    // Names of columns used by model\n    public static final String[] NAMES = new String[] {\n            "Bias",\n            "MaxWindPeriod",\n            "ChangeWindDirect",\n            "PressureChange",\n            "ChangeTempMag",\n            "EvapMM",\n            "MaxWindSpeed",\n            "Temp9am",\n            "RelHumid9am",\n            "Cloud9am",\n            "WindSpeed9am",\n            "Pressure9am",\n            "Temp3pm",\n            "RelHumid3pm",\n            "Cloud3pm",\n            "WindSpeed3pm",\n            "Pressure3pm",\n            "RainToday",\n            "TempRange"\n    };\n\n    // Derived features (we calculate ourselves in score0 implementation)\n    private static final String[] CALCULATED = new String[] {\n            "ChangeTemp",\n            "ChangeTempDir"\n    };\n\n    // Column domains, null means column is numerical\n    public static final String[][] DOMAINS = new String[][] {\n            /* Bias */ null,\n            /* MaxWindPeriod */ {"NA", "earlyAM", "earlyPM", "lateAM", "latePM"},\n            /* ChangeWindDirect */ {"c", "l", "n", "s"},\n            /* PressureChange */ {"down", "steady", "up"},\n            /* ChangeTempMag */ {"large", "small"},\n            /* EvapMM */ null,\n            /* MaxWindSpeed */ null,\n            /* Temp9am */ null,\n            /* RelHumid9am */ null,\n            /* Cloud9am */ null,\n            /* WindSpeed9am */ null,\n            /* Pressure9am */ null,\n            /* Temp3pm */ null,\n            /* RelHumid3pm */ null,\n            /* Cloud3pm */ null,\n            /* WindSpeed3pm */ null,\n            /* Pressure3pm */ null,\n            /* RainToday */ null,\n            /* TempRange */ null,\n            /* RISK_MM */ null\n    };\n\n    private final GenModel glm;\n    private final GenModel gbm;\n\n    // for each sub-model, mapping of the main model input and of the calculated columns to the sub-model input\n    private final Map<String, int[]> mappings;\n\n    // map of feature names to feature indices in the input array\n    private final Map<String, Integer> featureMap;\n\n    /**\n     * POJO constructor, creates instances of the sub-models and initializes\n     * helper structures for mapping input schema to the submodel schemas (mapping)\n     * and creates a map of feature names to indices to make value-lookups in code more readable. \n     */\n    public %s() { \n        super(NAMES, DOMAINS, "RISK_MM"); // response name goes here\n        glm = new %s();\n        gbm = new %s();\n        mappings = makeMappings(glm, gbm);\n        featureMap = new HashMap<>(NAMES.length);\n        for (int i = 0; i < NAMES.length; i++) {\n            featureMap.put(NAMES[i], i);\n        }\n    }\n\n    @Override\n    public String getUUID() { return "MyComplexPojo1"; } // just to show there can be anything here\n\n    // Important to override - BUG in POJO import for regression, will not work without this - FIXME\n    @Override\n    public int getNumResponseClasses() {\n        return 1;\n    }\n\n    @Override\n    public final double[] score0(double[] data, double[] preds) {\n        // (1) Show how to create derived feature (one numerical, the other one categorical)\n        // ChangeTemp = Temp3pm - Temp9am\n        double changeTemp = fNum("Temp3pm", data) - fNum("Temp9am", data);\n        double changeTempDir = changeTemp >= 0 ? 1 : 0; // changeTempDir is categorical: 0 == "down", 1 == "up"\n        double[] calculated = {\n                changeTemp,\n                changeTempDir\n        };\n\n        // (2) Show how to score multiple models\n        double[] glmPreds = score0SubModel(glm, data, calculated);\n        double[] gbmPreds = score0SubModel(gbm, data, calculated);\n\n        // (3) Show how to make decisions based on availability of an input (NA handling)\n        double bias = fNum("Bias", data);\n        if (!isNA(bias)) { // defined\n            // (4) Show to plug in a custom formula\n            preds[0] = glmPreds[0] * bias + (1 - bias) * gbmPreds[0];\n        } else {\n            String changeWindDirect = fCat("ChangeWindDirect", data);\n            // (5) Show how to return default values\n            if (isNA(changeWindDirect)) { // NA case, use default prediction\n                preds[0] = 1;\n            } else { // non-NA case, plug-in a formula based on categorical value\n                // (6) Show how to handle decisions based on categorical variable (different segments)\n                switch (changeWindDirect) {\n                    case "c":\n                    case "l":\n                        preds[0] = glmPreds[0] * 2;\n                        break;\n                    case "n":\n                        preds[0] = (glmPreds[0] + gbmPreds[0]) / 2;\n                        break;\n                    case "s":\n                        preds[0] = gbmPreds[0];\n                        break;\n                    default:\n                        preds[0] = -1;\n                }\n            }\n        }\n        return preds;\n    }\n\n    private static boolean isNA(double val) {\n        return Double.isNaN(val);\n    }\n\n    private static boolean isNA(String val) {\n        return val == null;\n    }\n\n    private double fNum(String feature, double[] data) {\n        Integer idx = featureMap.get(feature);\n        if (idx == null)\n            throw new IllegalArgumentException("Column \'" + feature + "\' is not part of model features.");\n        return data[idx];\n    }\n\n    private String fCat(String feature, double[] data) {\n        Integer idx = featureMap.get(feature);\n        if (idx == null)\n            throw new IllegalArgumentException("Column \'" + feature + "\' is not part of model features.");\n        if (Double.isNaN(data[idx]))\n            return null;\n        int level = (int) data[idx];\n        return DOMAINS[idx][level];\n    }\n\n    /**\n     * Scores a given sub-model - input is the original input row and also the calculated features.\n     * Input and calculated feature are mapped to the input of the sub-model.  \n     */\n    private double[] score0SubModel(GenModel model, double[] data, double[] calculated) {\n        int[] mapping = mappings.get(model.getUUID());\n        double[] subModelData = makeModelInput(data, calculated, mapping);\n        double[] subModelPreds = new double[model.getPredsSize()];\n        return model.score0(subModelData, subModelPreds);\n    }\n\n    private Map<String, int[]> makeMappings(GenModel... models) {\n        Map<String, int[]> mappings = new HashMap<>();\n        for (GenModel model : models) {\n            int[] mapping = mapInputNamesToModelNames(model);\n            mappings.put(model.getUUID(), mapping);\n        }\n        return mappings;\n    }\n\n    private static double[] makeModelInput(double[] data, double[] calculated, int[] mapping) {\n        double[] input = new double[mapping.length];\n        for (int i = 0; i < input.length; i++) {\n            int p = mapping[i];\n            if (p >= 0) {\n                input[i] = data[p];\n            } else {\n                input[i] = calculated[-p - 1];\n            }\n        }\n        return input;\n    }\n    \n    private int[] mapInputNamesToModelNames(GenModel subModel) {\n        int[] map = new int[subModel.nfeatures()];\n        for (int i = 0; i < map.length; i++) {\n            String name = subModel._names[i];\n            int p = indexOf(NAMES, name);\n            if (p < 0) {\n                p = indexOf(CALCULATED, name);\n                assert p >= 0 : "\'" + name + "\' needs to be one of the sub-model features or be a calculated feature.";\n                p = -p - 1;\n            }\n            map[i] = p;\n        }\n        return map;\n    }\n    \n    private static int indexOf(String[] ar, String element) {\n        for (int i = 0; i < ar.length; i++) {\n            if (ar[i].equals(element))\n                return i;\n        }\n        return -1;\n    }\n\n}\n\n// ===GLM===\n\n%s\n\n// ===GBM===\n\n%s\n'

def combined_pojo_class_name(glm_model, gbm_model):
    if False:
        return 10
    return 'Combined_' + glm_model.model_id + '_' + gbm_model.model_id

def generate_combined_pojo(glm_model, gbm_model):
    if False:
        for i in range(10):
            print('nop')
    glm_pojo_src = get_embeddable_pojo_source(glm_model)
    gbm_pojo_src = get_embeddable_pojo_source(gbm_model)
    results_dir = pyunit_utils.locate('results')
    combined_pojo_name = combined_pojo_class_name(glm_model, gbm_model)
    combined_pojo_path = os.path.join(results_dir, combined_pojo_name + '.java')
    combined_pojo_src = TEMPLATE % (combined_pojo_name, combined_pojo_name, glm_model.model_id, gbm_model.model_id, glm_pojo_src, gbm_pojo_src)
    with open(combined_pojo_path, 'w') as combined_file:
        combined_file.write(combined_pojo_src)
    return (combined_pojo_name, combined_pojo_path)

def get_embeddable_pojo_source(model):
    if False:
        while True:
            i = 10
    pojo_path = model.download_pojo(path=os.path.join(pyunit_utils.locate('results'), model.model_id + '.java'))
    return make_pojo_embeddable(pojo_path)

def make_pojo_embeddable(pojo_path):
    if False:
        for i in range(10):
            print('nop')
    pojo_lines = []
    with open(pojo_path, 'r') as pojo_file:
        pojo_lines = pojo_file.readlines()
        class_idx = next(filter(lambda idx: pojo_lines[idx].startswith('public class'), range(len(pojo_lines))))
        pojo_lines[class_idx] = pojo_lines[class_idx].replace('public class', 'class')
        pojo_lines = pojo_lines[class_idx - 1:]
    return ''.join(pojo_lines)

def generate_and_import_combined_pojo():
    if False:
        return 10
    if sys.version_info[0] < 3:
        print('This example needs Python 3.x+')
        return
    weather_orig = h2o.import_file(path=pyunit_utils.locate('smalldata/junit/weather.csv'))
    weather = weather_orig
    features = list(set(weather.names) - {'Date', 'RainTomorrow', 'Sunshine'})
    features.sort()
    response = 'RISK_MM'
    glm_model = H2OGeneralizedLinearEstimator()
    glm_model.train(x=features, y=response, training_frame=weather)
    glm_preds = glm_model.predict(weather)
    gbm_model = H2OGradientBoostingEstimator(ntrees=5)
    gbm_model.train(x=features, y=response, training_frame=weather)
    gbm_preds = gbm_model.predict(weather)
    weather = weather.drop('ChangeTemp')
    weather = weather.drop('ChangeTempDir')
    (combined_pojo_name, combined_pojo_path) = generate_combined_pojo(glm_model, gbm_model)
    print('Combined POJO was stored in: ' + combined_pojo_path)
    pojo_model = h2o.upload_mojo(combined_pojo_path, model_id=combined_pojo_name)
    weather['Bias'] = 1
    pojo_glm_preds = pojo_model.predict(weather)
    assert_frame_equal(pojo_glm_preds.as_data_frame(), glm_preds.as_data_frame())
    weather['Bias'] = 0
    pojo_gbm_preds = pojo_model.predict(weather)
    assert_frame_equal(pojo_gbm_preds.as_data_frame(), gbm_preds.as_data_frame())
    weather['Bias'] = float('NaN')
    for change_wind_dir in weather['ChangeWindDirect'].levels()[0]:
        weather_cwd = weather[weather['ChangeWindDirect'] == change_wind_dir]
        weather_orig_cwd = weather_orig[weather_orig['ChangeWindDirect'] == change_wind_dir]
        pojo_weather_cwd_preds = pojo_model.predict(weather_cwd)
        if change_wind_dir == 'c' or change_wind_dir == 'l':
            expected = glm_model.predict(weather_orig_cwd) * 2
            assert_frame_equal(pojo_weather_cwd_preds.as_data_frame(), expected.as_data_frame())
        elif change_wind_dir == 'n':
            expected = (glm_model.predict(weather_orig_cwd) + gbm_model.predict(weather_orig_cwd)) / 2
            assert_frame_equal(pojo_weather_cwd_preds.as_data_frame(), expected.as_data_frame())
        elif change_wind_dir == 's':
            expected = gbm_model.predict(weather_orig_cwd)
            assert_frame_equal(pojo_weather_cwd_preds.as_data_frame(), expected.as_data_frame())
if __name__ == '__main__':
    pyunit_utils.standalone_test(generate_and_import_combined_pojo)
else:
    generate_and_import_combined_pojo()