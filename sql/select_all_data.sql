select 
    CAST(open_time AS timestamp(3)) as open_time
    , open
    , asset
from quarterly_raw_data
order by open_time asc
limit <LIMIT>
;