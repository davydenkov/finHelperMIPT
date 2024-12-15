dict_new = {
    "Сталь": ['CHMF', 'MAGN', 'NLMK'],
    "Золото": ['POLY', 'PLZL', 'SELG', 'LNZL'],
    "Уголь": ['MTLR', 'RASP'],
    "Газ": ['GAZP', 'ROSN', 'LKOH', 'NVTK', 'TATN', 'BANE', 'SNGS', 'SNGSP'],
    "Земля": ['PHOR', 'AKRN', 'NKNC'],
    "Айти": ['YNDX', 'VKCO', 'OZON', 'QIWI'],
    "Банки": ['SBER', 'SBERP', 'TCSG', 'VTBR'],
    "Телеком": ['MTSS', 'RTKM', 'VEON-RX'],
    "Продукты": ['MGNT', 'FIVE', 'DSKY'],
    "Стройка": ['PIKK', 'ETLN', 'SMLT']
}


# вывод таблицы с всеми парами заданных акций
def final_table():
    # уставновка отношений между выбранными парами акций
    def find_ticker_relation(name):

        # сбор котировок акции для парного трейдинга
        def price_2(ticker):
            TICKs = [ticker]

            process = 0
            with requests.Session() as session:
                for TICK in TICKs:
                    data = apimoex.get_board_history(session, TICK)
                    if data == []:
                        continue
                    df = pd.DataFrame(data)
                    df = df[['BOARDID', 'TRADEDATE', 'CLOSE']]

            return df

        df_update = pd.DataFrame()

        for i in dict_new[name]:
            df = pd.DataFrame(price_2(i))
            df['BOARDID'] = i
            df_update = pd.concat([df_update, df])

        data = df_update.pivot(index='TRADEDATE',
                               columns='BOARDID',
                               values='CLOSE')
        data = data.dropna()
        data = data.tail(200)

        for i in data.columns:
            for k in data.columns:
                if i == k:
                    next
                else:
                    data[i + '/' + k] = data[i] / data[k]

        n = data.columns.to_list()
        itog = []

        for i in n:
            if (len(i) == 9) | (len(i) == 10):
                itog.append(i)

        data = data[itog]

        df_1 = data.describe().round(3).reset_index()
        df_1 = df_1[(df_1['index'] == 'mean') | (df_1['index'] == 'std')]
        df_1 = df_1.pivot_table(columns='index').reset_index()

        df_2 = data.tail(1).round(3).reset_index()
        df_2 = df_2.pivot_table(columns='index').reset_index()

        result = pd.merge(df_1, df_2, on="BOARDID")
        result['Result'] = result['index'] - result['mean']
        result['abs'] = [abs(i) for i in result['Result']]
        result['Отклонение'] = np.where(result['abs'] > result['std'], 'Да', 'Нет')
        result = result[result['Отклонение'] == 'Да']
        result['x'] = result['abs'] / result['std']
        result = result[result['x'] > 2]

        result = result.sort_values(by=['x'], ascending=False)

        result = result[result['Result'] > 0]

        result = result[result.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

        result = result.head(2)

        return result

    df_update = pd.DataFrame()

    for i in ["Сталь", "Золото", "Уголь", "Газ", "Земля", "Айти", "Банки", "Телеком", "Продукты", "Стройка"]:
        df = pd.DataFrame(find_ticker_relation(i))
        df['Рынок Аномалии'] = i
        df_update = pd.concat([df_update, df])

    df_update = df_update.reset_index(drop=True)

    result = df_update
    new_first = [i[:4] for i in result['BOARDID']]
    result['TICK'] = new_first

    table = result[['BOARDID', 'mean', 'index', 'x', 'TICK', 'Рынок Аномалии']]
    table['x'] = table['x'].astype('float')
    table = table.round(2)
    table.rename(columns={
        'TICK': 'Акция',
        'BOARDID': 'Пара',
        'mean': 'Ср. соотношение',
        'index': 'Тек. соотношение',
        'x': 'Превышение STD'
    },
        inplace=True)

    table = table.reset_index(drop=True)
    table = table.astype('string')

table['№ Пара | Тек.a/b | Ср.a/b | Акция | Рынок'] = table[
                                                             'Пара'] + ' | ' + \
                                                         table[
                                                             'Тек. соотношение'] + ' | ' + \
                                                         table[
                                                             'Ср. соотношение'] + ' | ' + \
                                                         table[
                                                             'Акция'] + ' | ' + \
                                                         table[
                                                             'Рынок Аномалии']

    table = table[['№ Пара | Тек.a/b | Ср.a/b | Акция | Рынок']]
    table[
        '№ Пара | Тек.a/b | Ср.a/b | Акция | Рынок'] = normalize_text(
        table, '№ Пара | Тек.a/b | Ср.a/b | Акция | Рынок')
    table.index = np.arange(1, len(table) + 1)

    table.to_excel('Парные аномалии.xlsx', index=False)

