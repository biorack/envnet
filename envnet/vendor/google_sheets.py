"""Vendored from metatlas.untargeted.tools"""

import pandas as pd  # Add this line
from io import StringIO


def get_google_sheet(notebook_name = "Sheet name",
                     token='/global/cfs/cdirs/metatlas/projects/google_sheets_auth/ipython to sheets demo-9140f8697062.json',
                     sheet_name = 'Sheet1',
                    literal_cols=None):
    """
    Returns a pandas data frame from the google sheet.
    Assumes header row is first row.

    To use the token hard coded in the token field,
    the sheet must be shared with:
    metatlas-ipython-nersc@ipython-to-sheets-demo.iam.gserviceaccount.com
    Unique sheet names are a requirement of this approach.

    """
    import gspread
    # from oauth2client.client import SignedJwtAssertionCredentials
    from oauth2client.service_account import ServiceAccountCredentials
#     scope = ['https://spreadsheets.google.com/feeds']
#     scope = ['https://www.googleapis.com/auth/spreadsheets']
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    #this is deprecated as of january, but we have pinned the version of oauth2.
    #see https://github.com/google/oauth2client/issues/401
#     json_key = json.load(open(token))
#     credentials = SignedJwtAssertionCredentials(json_key['client_email'], json_key['private_key'].encode(), scope)
    credentials = ServiceAccountCredentials.from_json_keyfile_name(token, scope)
    #here is the new way incase the version pin is removed
    #credentials = ServiceAccountCredentials.from_json_keyfile_name(token, scope)

    gc = gspread.authorize(credentials)
    wks = gc.open(notebook_name)
    istd_qc_data = wks.worksheet(sheet_name).get_all_values()
    headers = istd_qc_data.pop(0)
    df = pd.DataFrame(istd_qc_data,columns=headers)

    # Use round trip through read_csv to infer dtypes
    s = StringIO()
    df.to_csv(s)
    df2 = pd.read_csv(StringIO(s.getvalue()))
    if 'Unnamed: 0' in df2.columns:
        df2.drop(columns=['Unnamed: 0'],inplace=True)

    #turn list elements into lists instead of strings
    if literal_cols is not None:
        for col in literal_cols:
            df2[col] = df2[col].apply(literal_eval)
    df2 = df2.fillna('')

    return df2