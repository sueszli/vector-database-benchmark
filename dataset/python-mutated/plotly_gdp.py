"""
A demo with geo data shown with plotly.
"""
from flexx import flx
records = '\nAlbania,13.40,ALB\nAndorra,4.80,AND\nArmenia,10.88,ARM\nAustria,436.10,AUT\nBelgium,527.80,BEL\nBosnia and Herzegovina,19.55,BIH\nBulgaria,55.08,BGR\nCroatia,57.18,HRV\nCyprus,21.34,CYP\nCzech Republic,205.60,CZE\nDenmark,347.20,DNK\nEstonia,26.36,EST\nFinland,276.30,FIN\nFrance,2902.00,FRA\nUnited Kingdom,2848.00,GBR\nGeorgia,16.13,GEO\nGermany,3820.00,DEU\nGreece,246.40,GRC\nHungary,129.70,HUN\nIreland,245.80,IRL\nItaly,2129.00,ITA\nJordan,36.55,JOR\nKosovo,5.99,KSV\nKuwait,179.30,KWT\nLatvia,32.82,LVA\nLuxembourg,63.93,LUX\nMalta,10.57,MLT\nMoldova,7.74,MDA\nMonaco,6.06,MCO\nMongolia,11.73,MNG\nNetherlands,880.40,NLD\nNorway,511.60,NOR\nPoland,552.20,POL\nPortugal,228.20,PRT\nRomania,199.00,ROU\nSlovakia,99.75,SVK\nSlovenia,49.93,SVN\nSpain,1400.00,ESP\nSweden,559.10,SWE\nSwitzerland,679.00,CHE\nUkraine,134.90,UKR\n'
country_names = []
country_codes = []
country_gdps = []
for line in records.strip().splitlines():
    (name, gdp, code) = line.split(',')
    country_names.append(name)
    country_codes.append(code)
    country_gdps.append(float(gdp))
data = [{'type': 'scattergeo', 'mode': 'markers', 'locations': country_codes, 'marker': {'size': [v ** 0.5 for v in country_gdps], 'color': country_gdps, 'cmin': 0, 'cmax': 2000, 'colorscale': 'Viridis', 'colorbar': {'title': 'GDP'}, 'line': {'color': 'black'}}, 'name': 'Europe GDP'}]
layout = {'geo': {'scope': 'europe', 'resolution': 50}}

class PlotlyGeoDemo(flx.HBox):

    def init(self):
        if False:
            return 10
        flx.PlotlyWidget(data=data, layout=layout)
if __name__ == '__main__':
    flx.launch(PlotlyGeoDemo)
    flx.run()