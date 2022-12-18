function CNTRY_INDX(country)
    cntrlst = ["AUS", "AUT", "BEL", "BRA", "CAN", "CHN", "CZE", "DEU", "DNK", "ESP", "FIN", "FRA", "GBR", "GRC", "HUN", "IDN", "IND", "IRL", "ITA", "JPN", "KOR", "MEX", "NLD", "POL", "PRT", "ROM", "RUS", "SVK", "SVN", "SWE", "TUR", "TWN", "USA", "RoW"]
    indx = 0
    for i in eachindex(cntrlst)
        if cntrlst[i] == country
            indx = i
            break
        else
        end
    end
    if indx == 0
        return "Error: country not found"
    else
        return indx, cntrlst
    end
end