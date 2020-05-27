# cust_rg_palette = ['#FF3333', '#FF6666', '#FF9999', '#FFCCCC', '#CCFFCC', '#99FF99', '#66FF66', '#33FF33']
cust_rg_palette = tuple(
    reversed(['#FF3333', '#FF6666', '#FF9999', '#FFCCCC', '#CCFFCC', '#99FF99', '#66FF66', '#33FF33']))
cust_tooltip_ms = """
<div style="background: transparent; font-size: 15px; width:170px;">
<span><mark style="background-color:transparent; font-weight:bold;color:#003566;">Onset Cases:</mark>
<mark style="background-color:transparent; color:black;"> @{daily new cases ma}</mark></span>
<span><mark style="background-color:transparent; font-weight:bold;color:#003566;">Confirmed Cases:</mark>
<mark style="background-color:transparent; color:black;"> @{Confirmed New Cases}</mark></span>
</div>
"""
cust_tooltip_p_start = f"""
<div style="background: transparent; font-size: 15px;color: #003566; width:min-content;">
<span style="font-weight:bold;">
"""
cust_tooltip_p_end = "</span></div>"
d_col_width = 50
autocomplete_input_code = """
// handling rtplot separately from other plots due to patch complexity etc.
console.log('Patch sources value is:' + patchsources[choices.value]);
for (var i = 0; i < rtplot.renderers.length; i++) {
    rtplot.renderers[i].visible = true;
}
rtxaxis.visible = true;
rtyaxis.visible = true;
rtplot.y_range.end = patchsources[choices.value][3];
rtxaxis.bounds = [patchsources[choices.value][1], patchsources[choices.value][2]];
rtplot.renderers[0].data_source.data = patchsources[choices.value][0].data;
ms_plot_xaxis.visible = true;
ms_plot_yaxis0.visible = true;
for (var i = 0; i < ms_plot.renderers.length; i++) {
    ms_plot.renderers[i].visible = true;
}
//handle other plots
for (var p = 0; p < plots.length; p++) {
    xaxes[p].visible = true;
    yaxes[p].visible = true;
    for (var i = 0; i < plots[p].renderers.length; i++) {
        plots[p].renderers[i].visible = true;
    }
}
source.change.emit();
"""
cdsview_jsfilter_code = """
const indices = []
for (var i = 0; i <= source.data['name'].length; i++) {
    if (source.data['name'][i] == choices.value) {
        indices.push(i)
    }
}
return indices
"""
tblcallback_code = """
var selected_index = countytable_cds.selected.indices[0];
var selected_name = countytable_cds.data['name'][selected_index];
console.log('Selected county is ' + selected_name);
// handling rtplot separately from other plots due to patch complexity etc.
for (var i = 0; i < rtplot.renderers.length; i++) {
    rtplot.renderers[i].visible = true;
}
rtxaxis.visible = true;
rtyaxis.visible = true;
rtplot.y_range.end = patchsources[selected_name][3];
rtxaxis.bounds = [patchsources[selected_name][1], patchsources[selected_name][2]];
rtplot.renderers[0].data_source.data = patchsources[selected_name][0].data;
ms_plot_xaxis.visible = true;
ms_plot_yaxis0.visible = true;
for (var i = 0; i < ms_plot.renderers.length; i++) {
    ms_plot.renderers[i].visible = true;
}
//handle other plots
for (var p = 0; p < plots.length; p++) {
    //plots[p].title.text = selected_name;
    xaxes[p].visible = true;
    yaxes[p].visible = true;
    for (var i = 0; i < plots[p].renderers.length; i++) {
        plots[p].renderers[i].visible = true;
    }
}
choices.value = selected_name;
console.log('Patch sources value is:' + patchsources[selected_name]);
source.change.emit();
"""
autocomplete_in_tbl_code = """
var index_to_select = countytable_cds.data.name.indexOf(choices.value);
countytable_cds.selected.indices = [index_to_select];
countytable_cds.properties.selected.change.emit();
"""

# set large constants used on bokeh app construction
select_callback_code = """
console.log('Current data source is:' + select.value);
plot.renderers[0].data_source.data = cpleth_sources[select.value][0].data;
plot.visible = true;
plot.min_width = cpleth_sources[select.value][1][0];
plot.min_height = cpleth_sources[select.value][1][1];
plot.max_width = cpleth_sources[select.value][1][2];
plot.max_height = cpleth_sources[select.value][1][3];
var ttl = cb_obj.options.filter(option => option[0] == select.value)[0][1];
plot.title.text = ttl;
ttldiv.visible=true;
"""
full_names = ['Alabama', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Washington, DC',
              'Florida', 'Georgia', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine',
              'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
              'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
              'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin',
              'Wyoming']
base_dims = tuple((300, 200, 750, 500))  # min_width, min_height, max_width, max_height
cust_tooltip_above = """
<div style="background: transparent; font-size:15px;color:#003566;min-width:200px;">
    <div>
        <span style="font-weight:bold;">@name</span>
    </div>
    <div>
        <span>R<sub>t</sub>: @Rt{0.00}</span>
    </div>
    <div>
        <span>Total Estimated Cases: @total_estimated_cases</span>
    </div>
    <div>
        <span>Daily New Onset Cases: @daily_new_cases_ma</span>
    </div>
    <div>
        <span>2nd Order Growth: @2nd_order_growth{0.0}%</span>
    </div>
    <div>
        <span>Confirmed %Infected: @confirmed_infected{0.00}</span>
    </div>
</div>
"""

cust_tooltip_below = """
<div style="background: transparent; font-size: 15px;color: #003566; min-width:200px;">
    <div>
        <span style="font-weight:bold;">@name</span>
    </div>
    <div>
     <span>Infection density < threshold for R<sub>t</sub> calculation.</span>
    </div>
</div>
"""

custom_hover_code = """
var curRt = cb_data.renderer.data_source.data.Rt[cb_data.index.indices[0]];
if (isNaN(curRt)) {
    cb_obj.tooltips = cust_tooltip_below
    } else {
    cb_obj.tooltips = cust_tooltip_above
    }
"""
