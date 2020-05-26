# one-time manual preparation/munge of us census data to meet the reqs of this dashboard
# version used for all dashboard generation referenced in static_datasets/us_counties.tar.gz
import itertools
import os

import shapefile
import pandas as pd

import config


def get_map_data(stage_path: str):
    # munge census shapefile for dashboard use, modified based upon https://stackoverflow.com/a/42590806
    sfile = stage_path + "/cb_2019_us_county_5m.shp"
    dfile = stage_path + "/cb_2019_us_county_5m.dbf"
    shp = open(sfile, "rb")
    dbf = open(dfile, "rb")
    sf = shapefile.Reader(shp=shp, dbf=dbf)
    lats = []
    lons = []
    ct_name = []
    st_id = []
    ct_id = []
    for shprec in sf.shapeRecords():
        st_id.append(int(shprec.record[0]))
        ct_id.append(int(shprec.record[4]))
        ct_name.append(shprec.record[5])
        lon, lat = map(list, zip(*shprec.shape.points))
        indices = shprec.shape.parts.tolist()
        lat = [lat[i:j] + [float('NaN')] for i, j in zip(indices, indices[1:]+[None])]
        lon = [lon[i:j] + [float('NaN')] for i, j in zip(indices, indices[1:]+[None])]
        lat = list(itertools.chain.from_iterable(lat))
        lon = list(itertools.chain.from_iterable(lon))
        lats.append(lat)
        lons.append(lon)
    map_data = pd.DataFrame({'id': ct_id, 'lats': lats, 'lons': lons, 'state_id': st_id, 'county_name': ct_name})
    return map_data


def main() -> None:
    # census county boundary data source: https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_us_county_5m.zip
    unzipped_census_dir = f"{os.environ['HOME']}/Downloads/cb_2019_us_county_5m"
    map_output = get_map_data(unzipped_census_dir)
    state_fips_df = pd.read_csv(config.state_fips_csv)
    merged_df = pd.merge(map_output, state_fips_df, how='left', on='state_id')
    merged_df = merged_df.drop(columns=['state']).rename(columns={'abbr': 'state'})
    merged_df['name'] = merged_df.apply(lambda x: x['county_name'] + ', ' + str(x['state']).upper(), axis=1)
    merged_df.to_csv(config.us_counties_path, compression='gzip', index=False)


if __name__ == '__main__':
    main()