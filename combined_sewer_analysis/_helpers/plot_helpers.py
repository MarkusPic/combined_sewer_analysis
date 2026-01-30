__maintainer__ = "Markus Pichler"
__author__ = "Markus Pichler"
__copyright__ = "Copyright 2023, University of Technology Graz"
__credits__ = ["Markus Pichler"]
__license__ = "LGPL"
__version__ = "1.0.0"

from combined_sewer_analysis.date_analysis import DAY_KIND


def daykind_color(day, day_kind_detail=10):
    COLORS = {DAY_KIND.ALL_DAYS: '#ED1A52',  # TU ROT  # 'gray',  # '#377EB8'
              DAY_KIND.WORKDAY: '#377EB8',  # blue
              DAY_KIND.NO_WORKDAY: '#E41A1C',  # red
              DAY_KIND.WEEKEND: '#E41A1C',  # red
              DAY_KIND.SATURDAY: '#4DAF4A',  # red
              DAY_KIND.SUN_HOLIDAY: '#E41A1C',  # red
              DAY_KIND.HOLIDAY: '#4DAF4A',  # green
              DAY_KIND.FAKE_FRIDAY: '#222',
              DAY_KIND.BRIDGE_DAY: '#777'
    }

    if day_kind_detail >= 7:
        COLORS.update({'1 Monday': '#5E4FA2',
                       '2 Tuesday': '#3288BD',
                       '3 Wednesday': '#66C2A5',
                       '4 Thursday': '#ABDDA4',
                       '5 Friday': '#FDAE61',
                       '6 Saturday': '#F46D43',
                       '7 Sunday': '#D53E4F',
                       '8 Holiday': '#9E0142'}
                      )

        COLORS.update({'Monday': '#5E4FA2',
                       'Tuesday': '#3288BD',
                       'Wednesday': '#66C2A5',
                       'Thursday': '#ABDDA4',
                       'Friday': '#FDAE61',
                       'Saturday': '#F46D43',
                       'Sunday': '#D53E4F',
                       'Holiday': '#9E0142'}
                      )

    if day in COLORS:
        return COLORS[day]
    else:
        return '#888'


XLABEL_DIURNAL = 'Hours of the day'
