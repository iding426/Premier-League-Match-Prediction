"""
Static venue information for EPL teams.
Maps team names to stadium location (lat/lon), venue type, and city.
"""

# Venue mapping: team_name -> {city, lat, lon, venue_type}
# venue_type: 'open', 'dome', 'retractable'
VENUE_INFO = {
    # 2015-2024 EPL teams
    "Arsenal": {
        "stadium": "Emirates Stadium",
        "city": "London",
        "lat": 51.5549,
        "lon": -0.1084,
        "venue_type": "open",
    },
    "Aston Villa": {
        "stadium": "Villa Park",
        "city": "Birmingham",
        "lat": 52.5092,
        "lon": -1.8848,
        "venue_type": "open",
    },
    "Bournemouth": {
        "stadium": "Vitality Stadium",
        "city": "Bournemouth",
        "lat": 50.7352,
        "lon": -1.8383,
        "venue_type": "open",
    },
    "Brighton": {
        "stadium": "Amex Stadium",
        "city": "Brighton",
        "lat": 50.8609,
        "lon": -0.0831,
        "venue_type": "open",
    },
    "Burnley": {
        "stadium": "Turf Moor",
        "city": "Burnley",
        "lat": 53.7889,
        "lon": -2.2303,
        "venue_type": "open",
    },
    "Chelsea": {
        "stadium": "Stamford Bridge",
        "city": "London",
        "lat": 51.4817,
        "lon": -0.1910,
        "venue_type": "open",
    },
    "Crystal Palace": {
        "stadium": "Selhurst Park",
        "city": "London",
        "lat": 51.3983,
        "lon": -0.0854,
        "venue_type": "open",
    },
    "Everton": {
        "stadium": "Goodison Park",
        "city": "Liverpool",
        "lat": 53.4387,
        "lon": -2.9661,
        "venue_type": "open",
    },
    "Fulham": {
        "stadium": "Craven Cottage",
        "city": "London",
        "lat": 51.4749,
        "lon": -0.2217,
        "venue_type": "open",
    },
    "Huddersfield": {
        "stadium": "John Smith's Stadium",
        "city": "Huddersfield",
        "lat": 53.6542,
        "lon": -1.7684,
        "venue_type": "open",
    },
    "Hull": {
        "stadium": "MKM Stadium",
        "city": "Hull",
        "lat": 53.7461,
        "lon": -0.3677,
        "venue_type": "open",
    },
    "Leicester": {
        "stadium": "King Power Stadium",
        "city": "Leicester",
        "lat": 52.6204,
        "lon": -1.1420,
        "venue_type": "open",
    },
    "Liverpool": {
        "stadium": "Anfield",
        "city": "Liverpool",
        "lat": 53.4308,
        "lon": -2.9608,
        "venue_type": "open",
    },
    "Manchester City": {
        "stadium": "Etihad Stadium",
        "city": "Manchester",
        "lat": 53.4831,
        "lon": -2.2004,
        "venue_type": "open",
    },
    "Manchester United": {
        "stadium": "Old Trafford",
        "city": "Manchester",
        "lat": 53.4631,
        "lon": -2.2913,
        "venue_type": "open",
    },
    "Middlesbrough": {
        "stadium": "Riverside Stadium",
        "city": "Middlesbrough",
        "lat": 54.5781,
        "lon": -1.2168,
        "venue_type": "open",
    },
    "Newcastle United": {
        "stadium": "St James' Park",
        "city": "Newcastle",
        "lat": 54.9756,
        "lon": -1.6217,
        "venue_type": "open",
    },
    "Norwich": {
        "stadium": "Carrow Road",
        "city": "Norwich",
        "lat": 52.6221,
        "lon": 1.3089,
        "venue_type": "open",
    },
    "Nottingham Forest": {
        "stadium": "City Ground",
        "city": "Nottingham",
        "lat": 52.9400,
        "lon": -1.1328,
        "venue_type": "open",
    },
    "Sheffield United": {
        "stadium": "Bramall Lane",
        "city": "Sheffield",
        "lat": 53.3703,
        "lon": -1.4709,
        "venue_type": "open",
    },
    "Southampton": {
        "stadium": "St Mary's Stadium",
        "city": "Southampton",
        "lat": 50.9059,
        "lon": -1.3910,
        "venue_type": "open",
    },
    "Stoke": {
        "stadium": "bet365 Stadium",
        "city": "Stoke-on-Trent",
        "lat": 52.9883,
        "lon": -2.1753,
        "venue_type": "open",
    },
    "Sunderland": {
        "stadium": "Stadium of Light",
        "city": "Sunderland",
        "lat": 54.9144,
        "lon": -1.3882,
        "venue_type": "open",
    },
    "Swansea": {
        "stadium": "Swansea.com Stadium",
        "city": "Swansea",
        "lat": 51.6423,
        "lon": -3.9343,
        "venue_type": "open",
    },
    "Tottenham": {
        "stadium": "Tottenham Hotspur Stadium",
        "city": "London",
        "lat": 51.6043,
        "lon": -0.0665,
        "venue_type": "retractable", 
    },
    "Watford": {
        "stadium": "Vicarage Road",
        "city": "Watford",
        "lat": 51.6499,
        "lon": -0.4015,
        "venue_type": "open",
    },
    "West Bromwich Albion": {
        "stadium": "The Hawthorns",
        "city": "West Bromwich",
        "lat": 52.5092,
        "lon": -1.9642,
        "venue_type": "open",
    },
    "West Ham": {
        "stadium": "London Stadium",
        "city": "London",
        "lat": 51.5383,
        "lon": -0.0164,
        "venue_type": "retractable", 
    },
    "Wolverhampton": {
        "stadium": "Molineux Stadium",
        "city": "Wolverhampton",
        "lat": 52.5901,
        "lon": -2.1305,
        "venue_type": "open",
    },
    "Cardiff": {
        "stadium": "Cardiff City Stadium",
        "city": "Cardiff",
        "lat": 51.4726,
        "lon": -3.2030,
        "venue_type": "open",
    },
    "Leeds": {
        "stadium": "Elland Road",
        "city": "Leeds",
        "lat": 53.7779,
        "lon": -1.5720,
        "venue_type": "open",
    },
    "Brentford": {
        "stadium": "Gtech Community Stadium",
        "city": "London",
        "lat": 51.4908,
        "lon": -0.2889,
        "venue_type": "open",
    },
}


def get_venue_info(team_name):
    """
    Get venue information for a team.
    Returns dict with stadium, city, lat, lon, venue_type or None if not found.
    """
    return VENUE_INFO.get(team_name)


def get_venue_type_encoding(venue_type):
    """
    Encode venue type as numeric value.
    Returns: 0=open, 1=retractable, 2=dome
    """
    encoding = {
        "open": 0,
        "retractable": 1,
        "dome": 2,
    }
    return encoding.get(venue_type, 0)


def should_fetch_weather(venue_type):
    """
    Determine if weather should be fetched for this venue type.
    Dome venues don't need weather, open/retractable do.
    """
    return venue_type in ["open", "retractable"]
