-- customer_orders
with o as (
  select o.customer_id, o.product_id, coalesce(o.quantity, 0) as qty, p.category, p.unit_price
  from mini_shop.orders o join mini_shop.products p on o.product_id = p.product_id
), cat as (
  select customer_id, category, sum(qty) as units from o group by 1, 2
), fav as (
  select customer_id, category from (
    select customer_id, category,
           row_number() over (partition by customer_id order by units desc, category asc) as rn
    from cat) where rn = 1
)
select c.customer_id, c.name,
       count(o.product_id) as num_orders,
       round(coalesce(sum(o.qty * o.unit_price), 0), 2) as total_spent,
       f.category as favorite_category
from mini_shop.customers c
left join o on c.customer_id = o.customer_id
left join fav f on f.customer_id = c.customer_id
group by 1, 2, 5
order by 1

-- category_sales
with s as (
  select p.category,
         sum(coalesce(o.quantity, 0)) as units_sold,
         round(sum(coalesce(o.quantity, 0) * p.unit_price), 2) as revenue
  from mini_shop.products p left join mini_shop.orders o on o.product_id = p.product_id
  group by 1)
select category, units_sold, revenue,
       case when revenue > (select avg(revenue) from s) then 1 else 0 end as above_average
from s order by 1
