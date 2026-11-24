-- Reference (gold) solution. NEVER shown to the agent.
select
    city,
    count(*) as num_customers,
    round(sum(total_revenue), 2) as total_revenue
from {{ ref('customer_revenue') }}
group by city
