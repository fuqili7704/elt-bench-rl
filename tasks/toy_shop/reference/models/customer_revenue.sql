-- Reference (gold) solution. NEVER shown to the agent.
with completed as (
    select o.customer_id, o.quantity * p.price as amount
    from {{ source('raw', 'orders') }} o
    join {{ source('raw', 'products') }} p on o.product_id = p.product_id
    where o.status = 'completed'
)
select
    c.customer_id,
    c.name as customer_name,
    c.city,
    count(completed.customer_id) as order_count,
    round(coalesce(sum(completed.amount), 0), 2) as total_revenue
from {{ source('raw', 'customers') }} c
left join completed on c.customer_id = completed.customer_id
group by 1, 2, 3
