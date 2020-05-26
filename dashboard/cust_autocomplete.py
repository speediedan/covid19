from bokeh.core.properties import Bool
from bokeh.models import AutocompleteInput


class ExtAutocompleteInput(AutocompleteInput):
    __implementation__ = "cust_autocomplete.ts"
    case_sensitive = Bool(default=False, help="""Enable or disable case_sensitivity""")
