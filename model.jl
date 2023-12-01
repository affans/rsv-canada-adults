## RSV MODEL - CANADA/ADULTS 
# yang.ye@yale.edu 

using StatsBase, Random
using Distributions, DataFrames
using Logging, CSV, ProgressBars, IterTools
using Logging

@enum INFTYPE SUSC=0 NMA=1 OP=2 ED=3 GW=4 ICU=5 MV=6 DEATH=7
@enum VAXTYPE GSK=1 PFI=2
@enum MONTHS UNDEF=0 SEP=1 OCT=2 NOV=3 DEC=4 JAN=5 FEB=6 MAR=7 APR=8 MAY=9 JUN=10 JUL=11 AUG=12

# define an agent and all agent properties
Base.@kwdef mutable struct Human
    idx::Int64 = 0 
    agegroup::Int64 = 0    # in years. don't really need this but left it incase needed later
    comorbidity::Int64 = 0 # 0 no comorbidity, 1, 2, 3: comorbidities, 4: 4+ comorbidities
    dwelling::Int64 = 0    # 1 = LTCF, 2 = community
    rsvtype::INFTYPE = SUSC # 1 = outpatient, 2 = emergency, 3 = hospital
    rsvmonth::MONTHS = UNDEF  # month of RSV 
    rsvdays::Dict{String, Float64} = Dict("nma" => 0.0, "symp" => 0.0, "gw" => 0.0, "icu" => 0.0, "mv" => 0.0)
    vaccinated::Bool = false 
    vaxmonth::MONTHS = SEP  # default
    vaxtype::VAXTYPE = GSK # 1 
    vaxeff_op::Vector{Float64} = zeros(Float64, 24)
    vaxeff_ip::Vector{Float64} = zeros(Float64, 24)
    qalyweights::Dict{String, Float64} = Dict("norsv" => 0.0, "nma" => 0.0, "symp" => 0.0, "gw" => 0.0, "icu" => 0.0, "mv" => 0.0, "adverse" => 0.0)
end
Base.show(io::IO, ::MIME"text/plain", z::Human) = dump(z)

## system parameters
Base.@kwdef mutable struct ModelParameters    ## use @with_kw from Parameters
    popsize::Int64 = 100000
    numofsims::Int64 = 1000
    totalpopulation = 3_779_102 # from Seyed 
    vaccine_scenario::String = "S1" # S1 or S2
    vaccine_type::VAXTYPE = GSK # 0 - gsk, 1 pfizer 
    vaccine_profile::String = "fixed"   # or sigmoidal
    current_season::Int64 = 1 # either running 1 season or 2 seasons 
end

# constant variables
const humans = Array{Human}(undef, 0) 
const p = ModelParameters()  ## setup default parameters
pc(x) = Int(round(x / p.totalpopulation * p.popsize)) # convert to per-capita

function run_all_sims() 
    vtypes = [GSK, PFI]
    vscenarios = ["S1", "S2"]
    vprofiles = ["fixed", "sigmoidal"]
    for vtype in vtypes 
        for vscenario in vscenarios 
            for vprofile in vprofiles 
                p.vaccine_type = vtype 
                p.vaccine_scenario = vscenario 
                p.vaccine_profile = vprofile 
                #fname = "$(p.vaccine_type)_$(p.vaccine_scenario)_$(p.vaccine_profile).csv"
                #@info "running simulation for $(fname)"
                simulate()
            end
        end
    end 
end

function simulate() 
    #reset_params() # reset parameter values
    length(humans) == 0 && error("run reset_params() first")
    
    logger = Logging.NullLogger()
    global_logger(logger)
    Random.seed!(53)
   
    # create a dataframe with dynamically generated column names 
    # must equal the number of columns being returned from create_data_file() 
    _suff =  ["_ltcf", "_nonltcf"]
    create_title(p) = vec(map(x -> x[1] .* x[2], collect(product(p, _suff))))
    p1 = create_title(["vax"])
    p2 = create_title(["total_ma", "nma", "op", "ed", "totalhosp", "gw", "icu", "mv", "deaths"])
    p3 = create_title(["nmadays", "sympdays", "gwdays", "icudays", "mvdays"])
    p4 = create_title(["totalqalys", "qalyslost"])
    names = ["simid",  p1..., p2..., p3..., p4...]
        
    df_novax = DataFrame([name => [] for name in names])
    df_wivax = DataFrame([name => [] for name in names])

    # set up vaccine parameters 
    #p.vaccine_profile = "fixed"
    #p.vaccine_scenario = "S2"
    #p.vaccine_type = GSK

    fname_novax = "$(p.vaccine_type)_$(p.vaccine_scenario)_$(p.vaccine_profile)_novaccine.csv"
    fname_vax = "$(p.vaccine_type)_$(p.vaccine_scenario)_$(p.vaccine_profile)_wivaccine.csv"
    
    for sim in ProgressBar(1:1000)
        initialize_population() # initialize the population 
        initialize_vaccine() # initialize the vaccine efficacies
        
        p.current_season = 1 # set the default season

        # run through rsv simulation
        incidence() # sample incidence - OP, ED, HOSP, and death
        sample_days() # sample the number of days for each type of infection
        sample_qaly_weights() # sample the qaly weights for each infected person (could move right after incidence()

        # at this point save the data before running the vaccine function 
        _df = create_data_file() 
        map(x -> push!(df_novax, (sim, x...)), eachrow(_df)) # populate the dataframe 
        
        run_vaccine() # run the vaccine scenario -- will update numbers -- internally runs sample_days() 
        
        # save the data
        _df = create_data_file() 
        map(x -> push!(df_wivax, (sim, x...)), eachrow(_df)) # create the dataframe 
    end
    
    CSV.write("$(fname_novax)", df_novax)
    CSV.write("$(fname_vax)", df_wivax)
end

function create_data_file() 
    _vaccine_data = collect_vaccine() # 7x2
    _data_incidence = collect_incidence() # 7x18 (9)
    _data_infdays = collect_days() # 7x10 (5)
    _data_qalys  = calculate_qalys()
    _data = hcat(_vaccine_data, _data_incidence, _data_infdays, _data_qalys)
end

### HELPER FUNCTIONS 
function get_agegroup(x::Human) 
    # helper function to printout age group 
    if x.agegroup == 1 
        return "60 - 64"
    elseif x.agegroup == 2
        return "65 - 69"
    elseif x.agegroup == 3
        return "70 - 74"
    elseif x.agegroup == 4
        return "75 - 79"
    elseif x.agegroup == 5
        return "80 - 84"
    elseif x.agegroup == 6
        return "85+"
    end
end

function mth_indices() 
    # either returns [1, 2, 3, 4, 5, 6, 7, 8, 9] 
    # or [13, 14, 15, 16, 17, 18, 19, 20, 21]
    month_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9] .+ (12 * (p.current_season - 1))
end


### Iniltialization Functions 
reset_params() = reset_params(ModelParameters())
function reset_params(ip::ModelParameters)
    # the p is a global const
    # the ip is an incoming different instance of parameters 
    # copy the values from ip to p. 
    ip.popsize == 0 && error("no population size given")
    for x in propertynames(p)
        setfield!(p, x, getfield(ip, x))
    end
    # resize the human array to change population size
    resize!(humans, p.popsize)
    return
end

function initialize_population() 
    #Random.seed!(1234)
    # population per age group / Seyed / See StatsCanada 
    np_pop_distribution = [1004859, 855430, 702728, 523726, 342057, 350302]
    pop_distribution = pc.(np_pop_distribution)
    sf_agegroups = shuffle!(inverse_rle([1, 2, 3, 4, 5, 6], pop_distribution)) #
    push!(sf_agegroups, fill(6, abs(p.popsize - length(sf_agegroups)))...) # if length(sf_agegroups) != p.popsize # because of rounding issues
    
    @info "pop size per capita (from census data):" pop_distribution, sum(pop_distribution)
    @info "size of sf_agegroups: $(length(sf_agegroups))"

    # data for the number of comorbidity per age group
    co1 = Categorical([30, 26, 18, 12, 14] ./ 100)
    co2 = Categorical([21, 24, 20, 15, 20] ./ 100)
    co3 = Categorical([15, 21, 20, 17, 27] ./ 100)
    co4 = Categorical([10, 18, 20, 17, 35] ./ 100)
    co5 = Categorical([8, 14, 18, 19, 41] ./ 100)
    co6 = Categorical([5, 11, 17, 19, 48] ./ 100)
    co = [co1, co2, co3, co4, co5, co6]
    
    for i = 1:length(humans)
        humans[i] = Human() 
        x = humans[i] 
        x.idx = i  
        x.agegroup = sf_agegroups[i] 
        x.comorbidity = rand(co[x.agegroup]) - 1 # since categorical is 1-based. 
        x.dwelling = 2  # default is community        
    end

    # assign LTCF 
    total_ag2plus = sum(pop_distribution[2:end])
    ltcf_pop = 140000 / sum(np_pop_distribution[2:end]) * 100000
    prop_in_ltcf = ltcf_pop / total_ag2plus # 140,000 is the number of LTCF beds in Canada
    @info "total AG 2+ population: $total_ag2plus"
    @info "total LTCF: $ltcf_pop"
    @info "overall prop in LTCF: $prop_in_ltcf"
    
    _humans56 = findall(x -> x.agegroup in (5, 6), humans)
    _ltcf56 = round(Int, ltcf_pop * 0.76)
    @info "_ltcf56: $_ltcf56"
    _ltcfelig56 = shuffle!(sample(_humans56, _ltcf56, replace=false))

    _humans234 = findall(x -> x.agegroup in (2, 3, 4), humans)
    _ltcf234 = round(Int, ltcf_pop * (1 - 0.76))
    @info "_ltcf234: $_ltcf234"
    _ltcfelig234 = shuffle!(sample(_humans234, _ltcf234, replace=false))
    _ltcfelig = vcat(_ltcfelig56, _ltcfelig234)

    for x in _ltcfelig
        humans[x].dwelling = 1
    end

    @info "total in dwelling 1: $(length(findall(x -> x.dwelling == 1, humans)))"
    @info "total in dwelling 2: $(length(findall(x -> x.dwelling == 2, humans)))"

    return 
end

function initialize_vaccine() 
    #VAX_RNG = MersenneTwister(29948)
    # assign efficacy to each agent 
    # depends on the coverage value as well as the scenario 
    if p.vaccine_profile == "fixed" 
        gsk_outpatient = [0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.826, 0.776, 0.726, 0.676, 0.626, 0.576, 0.561, 0.561, 0.561, 0.561, 0.561, 0.561, 0.468, 0.374, 0.281, 0.187, 0.094, 0.000]
        gsk_inpatient = [0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.941, 0.885, 0.828, 0.772, 0.715, 0.659, 0.624, 0.624, 0.624, 0.624, 0.624, 0.624, 0.535, 0.428, 0.321, 0.214, 0.107, 0.000]
        pfi_outpatient = [0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.651, 0.619, 0.586, 0.554, 0.521, 0.489, 0.488, 0.488, 0.488, 0.488, 0.488, 0.488, 0.408, 0.326, 0.245, 0.163, 0.082, 0.000]
        pfi_inpatient = [0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.889, 0.868, 0.848, 0.827, 0.807, 0.786, 0.786, 0.786, 0.786, 0.786, 0.786, 0.786, 0.655, 0.524, 0.393, 0.262, 0.131, 0.000]
    else 
        gsk_outpatient = [0.825, 0.824, 0.823, 0.822, 0.820, 0.816, 0.812, 0.805, 0.795, 0.780, 0.759, 0.729, 0.689, 0.635, 0.569, 0.490, 0.403, 0.315, 0.233, 0.162, 0.104, 0.059, 0.026, 0.000]
        gsk_inpatient = [0.936, 0.934, 0.931, 0.927, 0.922, 0.914, 0.904, 0.891, 0.873, 0.849, 0.818, 0.778, 0.728, 0.668, 0.598, 0.519, 0.436, 0.352, 0.272, 0.198, 0.135, 0.081, 0.038, 0.000]
        pfi_outpatient = [0.649, 0.646, 0.642, 0.636, 0.630, 0.621, 0.610, 0.597, 0.580, 0.560, 0.536, 0.507, 0.474, 0.436, 0.393, 0.347, 0.299, 0.250, 0.202, 0.156, 0.113, 0.075, 0.041, 0.000]
        pfi_inpatient = [0.887, 0.886, 0.885, 0.883, 0.880, 0.876, 0.871, 0.863, 0.852, 0.836, 0.815, 0.787, 0.748, 0.699, 0.638, 0.564, 0.481, 0.393, 0.305, 0.222, 0.149, 0.087, 0.038, 0.000]
    end

    if p.vaccine_type == GSK 
        outpatient = gsk_outpatient 
        inpatient = gsk_inpatient
    elseif p.vaccine_type == PFI 
        outpatient = pfi_outpatient 
        inpatient = pfi_inpatient
    else 
        error("vaccine type incorrect")
    end
    
    # 90% of S1 are vaccinated in the first month
    S1_eligible = findall(x -> x.dwelling == 1 && rand() < 0.90, humans)
    S2_eligible = findall(x -> x.dwelling == 2 && rand() < 0.74, humans)
    @info "S1 eligible: $(length(S1_eligible))"
    @info "S2 eligible: $(length(S2_eligible))"

    if p.vaccine_scenario == "S1"  
        total_eligible = [S1_eligible...]
    elseif p.vaccine_scenario == "S2" 
        total_eligible = [S1_eligible..., S2_eligible...]
    else 
        error("vaccine scenario incorrect")
    end 
    
    s1_ctr = 0  
    s2_ctr = 0
    
    for idx in total_eligible 
        x = humans[idx] 
        x.vaccinated = true 
        x.vaxtype = p.vaccine_type 
        if x.dwelling == 1 
            s1_ctr += 1
            x.vaxmonth = SEP 
        else 
            s2_ctr += 1
            x.vaxmonth = rand() < 0.5 ? SEP : OCT            
        end
        x.vaxeff_op = roll_vaccine(Int(x.vaxmonth), outpatient)
        x.vaxeff_ip = roll_vaccine(Int(x.vaxmonth), inpatient)
    end 

    return s1_ctr, s2_ctr
end 

function roll_vaccine(month, efficacy)
    a = circshift(efficacy, month)
    a[1:month] .= 0
    return a
end

function agegroup_proportions()
    # calculate the proportion of each age group in the population 
    # we use this to distribute the sampled incidence (which is given only for 3 age groups)
    ag_sizes = [length(findall(x -> x.agegroup == i, humans)) for i = 1:6]
    _ag1 = ag_sizes[1] / (ag_sizes[1] + ag_sizes[2])
    _ag2 = ag_sizes[2] / (ag_sizes[1] + ag_sizes[2])
    _ag3 = ag_sizes[3] / (ag_sizes[3] + ag_sizes[4])
    _ag4 = ag_sizes[4] / (ag_sizes[3] + ag_sizes[4])
    _ag5 = ag_sizes[5] / (ag_sizes[5] + ag_sizes[6])
    _ag6 = ag_sizes[6] / (ag_sizes[5] + ag_sizes[6])

    return [_ag1, _ag2, _ag3, _ag4, _ag5, _ag6]
    
end

function _distribute_incidence(inftype, infmatrix)
    ctr = 0 
    for (ag, r) in enumerate(eachrow(infmatrix))  # each row is an age group (12, 34, 56)
        _cnt = sum(r)  # how many infected in that age group
        human_idx = shuffle!(findall(x -> x.agegroup == ag && x.rsvtype == SUSC, humans))[1:_cnt]
    
        for mth in 1:12 
            mcnt = r[mth] # how many infected in that month    
            for i in 1:mcnt 
                hidx = pop!(human_idx)
                x = humans[hidx] 
                x.rsvtype = inftype  
                x.rsvmonth = MONTHS(mth) # could also pop! tog
                ctr += 1 
            end 
        end
    end
    @info "made $ctr humans type: $inftype"
    return ctr
end

function _distribute_hosp(inftype, infmatrix, prop_ltcf) 
    
    ctr = 0
    for (ag, r) in enumerate(eachrow(infmatrix))  # each row is an age group: 12, 34, 56
        
        # create two groups of humans: one from LTCF and one from non-LTCF 
        # when selecting the human, pop! a human from the right group
        # this is only true for age groups 2, 3, 4, 5, 6

        if ag in (2, 3, 4, 5, 6) 
            _humans_non_ltcf = shuffle!(findall(x -> x.agegroup == ag && x.rsvtype == SUSC && x.dwelling == 2, humans))
            _humans_ltcf = shuffle!(findall(x -> x.agegroup == ag && x.rsvtype == SUSC && x.dwelling == 1, humans))
        else  
            # for ag1, could pick from LTCF or not... also, keep variable names the same so that the code below works 
            _humans = shuffle!(findall(x -> x.agegroup == ag && x.rsvtype == SUSC, humans)) # 
            _humans_non_ltcf = _humans
            _humans_ltcf = _humans
        end
        
        for mth in 1:12 
            mcnt = r[mth] # how many infected in that month    
            for i in 1:mcnt 
                if rand() < prop_ltcf 
                    humans_idx = _humans_ltcf 
                else 
                    humans_idx = _humans_non_ltcf 
                end
                hidx = pop!(humans_idx)
                x = humans[hidx] 
                x.rsvtype = inftype  
                x.rsvmonth = MONTHS(mth) # could also pop! tog
                ctr += 1 
            end 
        end
    end
    @info "made $ctr humans type: $inftype"
    return ctr
end

function incidence() 
    # medically attended (calculated using the per 100,000 incidence of each age group)
    # includes ED, OP, and HOSP

    # distribution of MA per month
    month_distr = [0.006607792, 0.008512851, 0.018152986, 0.055796604, 0.155275108, 0.200087753, 
                   0.225014937, 0.167146155, 0.0970358, 0.043135959, 0.015993756, 0.0072403]

    # sample incidence MA
    inc_ag12 = rand(410:905); inc_ag34 = rand(274:599); inc_ag56 = rand(172:365)
    sampled_incidence = [inc_ag12, inc_ag12, inc_ag34, inc_ag34, inc_ag56, inc_ag56] .* agegroup_proportions()
    seasonal_incidence = sampled_incidence * transpose(month_distr)
    
    # sample hospitalization 
    hosp_inc_from_data = rand(118:172) # From Open Form Infectious Diseases 2023 (canadian paper)
    #[0.18*hosp_inc_from_data, 0.18*hosp_inc_from_data, ]
    hosp_ag12 = 0.18*hosp_inc_from_data; hosp_ag34 = 0.26*hosp_inc_from_data; hosp_ag56 = 0.56*hosp_inc_from_data
    sampled_hospitalization = [hosp_ag12, hosp_ag12, hosp_ag34, hosp_ag34, hosp_ag56, hosp_ag56] .* agegroup_proportions()
    seasonal_hospitalization = round.(Int, sampled_hospitalization * transpose(month_distr))
    total_hosp = sum(seasonal_hospitalization) # different than sampled_hosp cuz of rounding but should be very close

    # proportion of overall incidence are ED
    seasonal_emergency = round.(Int, rand(Uniform(0.06, 0.09)) .* seasonal_incidence)

    # The left over are outpatient (OP = incidence - ED - HOSP)
    seasonal_outpatient = round.(Int, seasonal_incidence .- seasonal_emergency .- seasonal_hospitalization)
    
    # sample deaths Categorical([5.3, 12.5, 14.1, 25.7, 20.9, 21.5] ./ 100) # NOTE: ONLY FIVE AGE GROUPS
    # seasonal_death = [sum(rand(x) .< 0.061) for x in seasonal_hospitalization]
    seasonal_death = new_death_distribution(seasonal_hospitalization) # seasonal_death_distribute(seasonal_hospitalization)
    total_death = sum(seasonal_death)

    # sample the ICU/MV/GW
    #seasonal_icu_and_mv = [sum(rand(x) .< 0.135) for x in seasonal_hospitalization]
    seasonal_icu_and_mv = [sum(rand(x) .< 0.074) for x in seasonal_hospitalization] # it's actually 13.5% but 6.1% death sampled up there are assumed to be also part of ICU/MV (Their days are sampled as ICU/MV)
    
    # split between icu/mv/gw
    seasonal_mv = [sum(rand(x) .< 0.523) for x in seasonal_icu_and_mv]
    seasonal_icu = seasonal_icu_and_mv .- seasonal_mv
    seasonal_gw = seasonal_hospitalization .- seasonal_icu_and_mv .- seasonal_death  # this gives us the general ward 
    
    # print sampled incidence matrices 
    #@info "seasonal incidence" seasonal_incidence
    #@info "season outpatient" seasonal_outpatient
    #@info "seasonal emergency" seasonal_emergency
    @info "seasonal hospitalization" seasonal_hospitalization total_hosp
    @info "seasonal death" seasonal_death total_death
    @info "seasonal_icu" seasonal_icu_and_mv sum(seasonal_icu_and_mv)
    @info "seasonal_mv" seasonal_mv sum(seasonal_mv)
    @info "seasonal gw" seasonal_gw sum(seasonal_gw)
    
    # for each matrix, we need to distribute the infections over the humans
    # all_matrices = [seasonal_outpatient, seasonal_emergency, seasonal_gw, seasonal_icu, seasonal_mv]
    i1 = _distribute_incidence(OP, seasonal_outpatient)
    i2 = _distribute_incidence(ED, seasonal_emergency)
    prop_ltcf = rand(Uniform(0.08, 0.15))
    i3 = _distribute_hosp(GW, seasonal_gw, prop_ltcf)
    i4 = _distribute_hosp(ICU, seasonal_icu, prop_ltcf)
    i5 = _distribute_hosp(MV, seasonal_mv, prop_ltcf)
    i6 = _distribute_hosp(DEATH, seasonal_death, prop_ltcf)
    return  
end 

function new_death_distribution(seasonal_hosp) 

    death_probs = [7.6, 7.6, 8.1, 8.1, 14.0, 14.0] ./ 100 
    seasonal_death = zeros(Int64, 6, 12)
    for (ag, r) in enumerate(eachrow(seasonal_hosp))
        death_prob = death_probs[ag]

        for mth in 1:12 
            hosp_cnt = r[mth] 
            death_cnt = rand(hosp_cnt) .< death_prob
            seasonal_death[ag, mth] = sum(death_cnt)
        end
    end

    ## test the function 
    #prop = sum.(eachrow(seasonal_death))  ./ sum.(eachrow(seasonal_hosp)) .* 100
    #@info "seasonal death" seasonal_death
    #@info "proportion of death: $prop"
    # return prop 

    return seasonal_death
end

function seasonal_death_distribute(seasonal_hosp)
    # given total hospitalization stratified by agegroup, use MC to sample the number of deaths over the season
    ## OLD FUNCTION: NOT USED ANYMORE -- USING DEATH PROBABILITIES  

    total_hosp = sum(seasonal_hosp)
    total_death2 = round.(Int, 0.11 * total_hosp)
    death_probs = Categorical([5.3, 12.5, 14.1, 25.7, 20.9, 21.5] ./ 100)  # last two age groups split by seyed 
    death_in_ag = rand(death_probs, total_death2)
    cm = countmap(death_in_ag)
    #@info "total death / split over ag" total_death2 cm
    
    seasonal_death = zeros(Int64, 6, 12)
    for (ag, deathcnt) in cm  
        hvec = copy(seasonal_hosp[ag, :]) 
        dvec = zeros(Int64, 12) 

        for _ in Base.OneTo(deathcnt) # don't care about the actual value, just need to loop deathcnt amount of times
            pos_idx = findall(x -> x > 0, hvec)
            idx = rand(pos_idx) 
            dvec[idx] += 1
            hvec[idx] -= 1
        end
        seasonal_death[ag, :] .= dvec
    end
    return seasonal_death
end 

function sample_days() 
    # sample number of days for each type of infection 
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans)
    @info "sampling RSV days for $(length(all_sick)) humans"
    for idx in all_sick
        x = humans[idx]
        sample_inf_days_human(x)
    end
end

function sample_inf_days_human(x::Human) 
    rsvtype = x.rsvtype
    rsvtype == SUSC && error("can't sample RSV days for a SUSCEPTIBLE person id $(x.idx)")

    nmadays = 0.0 
    sympdays = 0.0 
    gwdays = 0.0 
    icudays = 0.0
    mvdays = 0.0

    if rsvtype == NMA 
        nmadays += rand(Uniform(2, 8)) 
    elseif rsvtype in (OP, ED) 
        sympdays += rand(Uniform(7, 14))
    elseif rsvtype == GW 
        sympdays += rand(Gamma(4.3266,0.9434))
        gwdays += rand(Gamma(3.0658, 3.4254))
    elseif rsvtype == ICU 
        sympdays += rand(Gamma(4.3266,0.9434))
        gwdays += rand(Gamma(2.0673, 1.2438)) 
        icudays += rand(Gamma(4.1049,1.2876))
        gwdays += rand(Gamma(1.1092, 11.2493)) # post ICU
    elseif rsvtype == MV 
        sympdays += rand(Gamma(4.3266,0.9434)) 
        gwdays += rand(Gamma(2.0673, 1.2438)) # pre ICU 
        icudays += rand(Gamma(1.5910,3.4570)) # in ICU but prior to moving to MV 
        mvdays += rand(Gamma(1.4306, 7.5620))  # mechanical ventilation
        gwdays += rand(Gamma(1.1092, 11.2493)) # post ICU
    elseif rsvtype == DEATH
        sympdays += rand(Gamma(4.3266,0.9434)) 
        gwdays += rand(Gamma(2.0673, 1.2438)) # pre ICU 
        if rand() < 0.523 # assume 52.3% of deaths are in MV, the rest in ICU
            icudays += rand(Gamma(1.5910,3.4570)) # in ICU but prior to moving to MV 
            mvdays += rand(Gamma(1.4306, 7.5620))  # mechanical ventilation
            # no post GW days because assume person death in MV/ICU
        else 
            icudays += rand(Gamma(4.1049,1.2876))
        end
    end

    x.rsvdays["nma"] = nmadays
    x.rsvdays["symp"] = sympdays
    x.rsvdays["gw"] = gwdays
    x.rsvdays["icu"] = icudays
    x.rsvdays["mv"] = mvdays
    return (nmadays, sympdays, gwdays, icudays, mvdays)
end


function sample_qaly_weights() 
    # sample the QALY weights for each sick agent object
    # - does not calculate the actual QALY loss. 
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans)
    @info "sampling QALYs for $(length(all_sick)) humans"
    
    # sample the NO RSV weight using beta distributions
    d1 = Beta(162.61, 48.57)
    d2 = Beta(141.60, 44.72)
    d3 = Beta(112.16, 39.41)
    d4 = Beta(76.03, 32.58)
    d5 = Beta(41.96, 24.64)
    d6 = Beta(33.08, 31.78)
    betas = [d1, d2, d3, d4, d5, d6]

    # multiplicative factors for each disease type
    nonma_wgt = 0.88
    op_wgt = 0.76 
    gw_wgt = 0.35 
    icu_wgt = 0.1 
    mv_wgt = 0.1
    
    for idx in all_sick
        x = humans[idx]
        ag = x.agegroup 
        
        # get the non-rsv distribution
        nonrsv_dist = betas[ag]
        norsv_wgt = rand(nonrsv_dist)
        
        x.qalyweights["norsv"] = norsv_wgt
        x.qalyweights["nma"] = norsv_wgt * nonma_wgt
        x.qalyweights["symp"] = norsv_wgt * op_wgt
        x.qalyweights["gw"] = norsv_wgt * gw_wgt
        x.qalyweights["icu"] = norsv_wgt * icu_wgt
        x.qalyweights["mv"] = norsv_wgt * mv_wgt
    end 
end

function calculate_qalys() 
    sd = map([1, 2]) do dw
        split_data = zeros(Float64, 7, 2)  # allocate return vector

        # go througḣall sick but non-dead people 
        all_sick = findall(x -> Int(x.rsvtype) > 0 && x.rsvtype != DEATH && x.dwelling == dw, humans)
        for i in all_sick
            x = humans[i]  
            ag = x.agegroup

            nma_days = x.rsvdays["nma"]
            symp_days = x.rsvdays["symp"]
            gw_days =  x.rsvdays["gw"] 
            icu_days = x.rsvdays["icu"]
            mv_days = x.rsvdays["mv"]
                    
            wgt_norsv = x.qalyweights["norsv"] 
            wgt_nma = x.qalyweights["nma"] 
            wgt_op = x.qalyweights["symp"]
            wgt_nonicu = x.qalyweights["gw"]
            wgt_icu = x.qalyweights["icu"] 
            wgt_mv = x.qalyweights["mv"] 
            
            non_rsv_days = 365 - (nma_days + symp_days + gw_days + icu_days + mv_days)
            totalqaly = wgt_norsv*non_rsv_days + wgt_nma*nma_days + wgt_op*symp_days + wgt_nonicu*gw_days + wgt_icu*icu_days + wgt_mv*mv_days 
            totalqaly = totalqaly / 365

            split_data[ag+1, 1] += totalqaly
        end
        split_data[1, 1] = sum(split_data[2:end, 1])  # sum up the qalys for each age group for the top row 
        
        all_dead = findall(x -> x.rsvtype == DEATH && x.dwelling == dw, humans) 
        # wgt_qalylost = [9.47, 7.79, 5.93, 4.49, 2.97, 1.49]
        wgt_qalylost = [10.4, 8.4, 6.3, 4.7, 3.1, 1.5]
        
        for i in all_dead
            x = humans[i] 
            ag = x.agegroup
            qalylost = wgt_qalylost[ag]
            split_data[ag+1, 2] += qalylost
        end
        split_data[1, 2] = sum(split_data[2:end, 2])  # sum up the qalys for each age group for the top row
        return split_data
    end
    return hcat(sd...)
end

function run_vaccine() 
    op_to_nma = 0 
    ip_to_op = 0
    all_sick = humans[findall(x -> Int(x.rsvtype) > 0, humans)] # find all sick individuals to calculate their outcomes
    for x in all_sick  
        rn = rand() 
        if x.rsvtype == OP || x.rsvtype == ED  # they become NMA 
            if rn < x.vaxeff_op[Int(x.rsvmonth)]
                x.rsvtype = NMA
                op_to_nma += 1
                sample_inf_days_human(x)
            end
        end 
        if x.rsvtype in (GW, ICU, MV, DEATH)
            if rn < x.vaxeff_ip[Int(x.rsvmonth)]
                x.rsvtype = OP
                ip_to_op += 1
                sample_inf_days_human(x)
            end
        end 
    end
    return (op_to_nma, ip_to_op)
end

function collect_incidence() 
    # for each dwelling type, collect the incidence data
    sd = map(1:2) do dw
        split_data = zeros(Float64, 7, 9) # 7 rows (everyone + 6 age groups) 
        for ag in 0:6 
            if ag == 0 
                _humans = humans[findall(x -> Int(x.rsvtype) > 0 && x.dwelling == dw, humans)]
            else 
                _humans = humans[findall(x -> Int(x.rsvtype) > 0 && x.agegroup == ag && x.dwelling == dw, humans)]
            end
            all_sick = length(findall(x -> Int(x.rsvtype) > 0, _humans))  
            nma = length(findall(x -> x.rsvtype == NMA, _humans))
            outpatients = length(findall(x -> x.rsvtype == OP, _humans))   
            emergency = length(findall(x -> x.rsvtype == ED, _humans))   
            gw = length(findall(x -> x.rsvtype == GW, _humans))
            totalicu = length(findall(x -> x.rsvtype == ICU, _humans))
            totalmv = length(findall(x -> x.rsvtype == MV, _humans))
            totaldeath = length(findall(x -> x.rsvtype == DEATH, _humans))
            allhospitalizations = gw + totalicu + totalmv + totaldeath
            split_data[ag+1, :] .= (all_sick, nma, outpatients, emergency, allhospitalizations, gw, totalicu, totalmv, totaldeath)
        end
        return split_data
    end
    return hcat(sd...)
end

function collect_days() 

    # for each dwelling type, collect the incidence data
    sd = map(1:2) do dw
        split_data = zeros(Float64, 7, 5) # 7 rows for all + 6 age groups 
        all_sick = findall(x -> Int(x.rsvtype) > 0 && x.dwelling == dw, humans)
        for idx in all_sick
            x = humans[idx]
            ag = x.agegroup
            nma = x.rsvdays["nma"]
            s = x.rsvdays["symp"]
            h =  x.rsvdays["gw"] 
            i_nmv = x.rsvdays["icu"]
            i_mv = x.rsvdays["mv"]
            split_data[1, :] .+= [nma, s, h, i_nmv, i_mv]
            split_data[ag+1, :] .+= [nma, s, h, i_nmv, i_mv]
        end
        return split_data 
    end
    return hcat(sd...)
end

function collect_incidence_ltcfstats() 
    _humans = findall(x -> Int(x.rsvtype) in (GW, ICU, MV, DEATH), humans)
    total_hosp = length(_humans)
    total_ltcf = 0 
    for x in _humans 
        if humans[x].dwelling == 1 
            total_ltcf += 1
        end
    end
    @info "total hospitalizations: $total_hosp"
    @info "total hospitalizations from LTCF: $total_ltcf"
    return total_ltcf
end

function collect_toi() 
    # function calculates the time of infection for each agent 
    # slight bug but not relevant -- if a person dies, their TOI is not really sampled and so this function returns UNDEF
    all_sick = findall(x -> Int(x.rsvtype) > 0, humans)
    split_data = zeros(Float64, 7, 1)
    toi = [humans[i].rsvmonth for i in all_sick]
    countmap(toi)
end

function collect_vaccine()
    split_data = zeros(Float64, 7, 2) # 7 rows for all + 6 age groups 
    
    
    h1_ltcf = length(findall(x -> x.vaccinated == true && x.dwelling==1 && x.agegroup == 1, humans))
    h2_ltcf = length(findall(x -> x.vaccinated == true && x.dwelling==1 && x.agegroup == 2, humans))
    h3_ltcf = length(findall(x -> x.vaccinated == true && x.dwelling==1 && x.agegroup == 3, humans))
    h4_ltcf = length(findall(x -> x.vaccinated == true && x.dwelling==1 && x.agegroup == 4, humans))
    h5_ltcf = length(findall(x -> x.vaccinated == true && x.dwelling==1 && x.agegroup == 5, humans))
    h6_ltcf = length(findall(x -> x.vaccinated == true && x.dwelling==1 && x.agegroup == 6, humans))
    ht_ltcf = h1_ltcf + h2_ltcf + h3_ltcf + h4_ltcf + h5_ltcf + h6_ltcf

    h1_nonltcf = length(findall(x -> x.vaccinated == true && x.dwelling==2 && x.agegroup == 1, humans))
    h2_nonltcf = length(findall(x -> x.vaccinated == true && x.dwelling==2 && x.agegroup == 2, humans))
    h3_nonltcf = length(findall(x -> x.vaccinated == true && x.dwelling==2 && x.agegroup == 3, humans))
    h4_nonltcf = length(findall(x -> x.vaccinated == true && x.dwelling==2 && x.agegroup == 4, humans))
    h5_nonltcf = length(findall(x -> x.vaccinated == true && x.dwelling==2 && x.agegroup == 5, humans))
    h6_nonltcf = length(findall(x -> x.vaccinated == true && x.dwelling==2 && x.agegroup == 6, humans))
    ht_nonltcf = h1_nonltcf + h2_nonltcf + h3_nonltcf + h4_nonltcf + h5_nonltcf + h6_nonltcf    
    
    [ht_ltcf; h1_ltcf; h2_ltcf; h3_ltcf; h4_ltcf; h5_ltcf; h6_ltcf;;
     ht_nonltcf; h1_nonltcf; h2_nonltcf; h3_nonltcf; h4_nonltcf; h5_nonltcf; h6_nonltcf]

end


function collect_ltcf_stats() 
    all_humans = length(findall(x -> x.agegroup in (2, 3, 4, 5, 6), humans))
    all_ltcf = length(findall(x -> x.dwelling == 1, humans))
    all_ag56 = length(findall(x -> x.dwelling == 1 && x.agegroup in (5, 6), humans))
    all_ag234 = length(findall(x -> x.dwelling == 1 && x.agegroup in (2, 3, 4), humans))
    @info "total number of AG 2+ humans: $all_humans"
    @info "total LTCF: $(all_ltcf)"
    @info "proportion of LTCF in 80+: $(all_ag56 / all_ltcf)"
    return 
end 

function test_ltcf_props() 
    ltcf_totals = zeros(Float64, 100)
    for sim = 1:100 
        println("sim: $sim")
        initialize_population() 
        _hmns = humans[findall(x -> x.rsvtype in (GW, ICU, MV, DEATH), humans)]
        @info "humans before incidence: $(length(_hmns))"
        incidence();
        _hmns = humans[findall(x -> x.rsvtype in (GW, ICU, MV, DEATH), humans)]
        total_hosp = length(_hmns)
        ltcf = 0 
        for x in _hmns
            x.dwelling == 1 && (ltcf += 1)
        end
        ltcf_totals[sim] = ltcf / total_hosp
    end

    #@info "testing ltcf props"
    #@info "total hosp+death: $total_hosp"
    @info "total ltcf (should be 10%) prop: $(mean(ltcf_totals))"
end

function test_vaccine_allocation() 
    h = findall(x -> x.vaccinated == true, humans) 
    h1 = findall(x -> x.vaccinated == true && x.dwelling==1, humans)
    h2 = findall(x -> x.vaccinated == true && x.dwelling==2, humans)
    h3 = findall(x -> x.vaccinated == true && x.dwelling==2 && x.vaxmonth == SEP, humans)
    h4 = findall(x -> x.vaccinated == true && x.dwelling==2 && x.vaxmonth == OCT, humans)

    @info "total vaccinated: $(length(h))" 
    @info "total vaccinated in LTCF: $(length(h1))"
    @info "total vaccinated in community: $(length(h2))"
    @info "total vaccinated in community (SEP): $(length(h3))"
    @info "total vaccinated in community (OCT): $(length(h4))"
end

function test_vaccine_efficacy() 
    h1 = humans[findall(x -> x.vaccinated == true && x.dwelling == 1, humans)]
    h2 = humans[findall(x -> x.vaccinated == true && x.dwelling == 2 && x.vaxmonth == SEP, humans)]
    h3 = humans[findall(x -> x.vaccinated == true && x.dwelling == 2 && x.vaxmonth == OCT, humans)]

    println("""
    LTCF:
        vaccinated? $(h1[1].vaccinated)
        vaxtype? $(h1[1].vaxtype)
        vaxmonth? $(h1[1].vaxmonth)
        vaxeff_op? $(h1[1].vaxeff_op)
        vaxeff_ip? $(h1[1].vaxeff_ip)
    """)
    println("""
    SEP NON LTCF:
        vaccinated? $(h2[1].vaccinated)
        vaxtype? $(h2[1].vaxtype)
        vaxmonth? $(h2[1].vaxmonth)
        vaxeff_op? $(h2[1].vaxeff_op)
        vaxeff_ip? $(h2[1].vaxeff_ip)
    """)
    println("""
    OCT NON LTCF
        vaccinated? $(h3[1].vaccinated)
        vaxtype? $(h3[1].vaxtype)
        vaxmonth? $(h3[1].vaxmonth)
        vaxeff_op? $(h3[1].vaxeff_op)
        vaxeff_ip? $(h3[1].vaxeff_ip)
    """)
end

function check_death()
    prop = zeros(Float64, 1000)
    for i = 1:1000
        d, h = incidence()
        prop[i] = d/h
    end
    return prop
end
