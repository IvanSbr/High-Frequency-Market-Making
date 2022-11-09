from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class BestPosStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                #place order
                bid_order = sim.place_order( receive_ts, 0.001, 'BID', best_bid )
                ask_order = sim.place_order( receive_ts, 0.001, 'ASK', best_ask )
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders


class StoikovStrategy:
    '''
        This strategy based on 'High-frequency trading in a limit order book' by MARCO AVELLANEDA and SASHA STOIKOV.

        I will use 'agent with infinite horizon' because crypto exchange works 24/7 
    '''

    def __init__(self, q: float, gamma: float, sigma: float, delay: float, hold_time:Optional[float] = None) -> None:
        '''
            Args:
                q: inventory 
                gamma: constant  
                sigma: constant
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.q = q
        self.gamma = gamma
        self.sigma = sigma
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        md_list:List[MdUpdate] = [] #market data list

        trades_list:List[OwnTrade] = [] #executed trades list

        updates_list = [] #all updates list

        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        prev_time = -np.inf #last order timestamp
        
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []

        q_min = 0
        q_max = 5000


        while True:

            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break

            #save updates
            updates_list += updates

            for update in updates:

                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)

                    if update.side == 'BID':
                        self.q += update.size
                    else:
                        self.q -= update.size

                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                
                s = (best_bid + best_ask) / 2 # best price 

                omega = 0.5 * self.gamma * self.gamma * self.sigma * self.sigma * (q_max + 1) * (q_max + 1)
                denomimator = 2 * omega - self.gamma * self.gamma * self.sigma * self.sigma * self.q * self.q

                r_a = s + 1/self.gamma * np.log(1 + ((1 - 2*self.q) * self.gamma * self.gamma * self.sigma * self.sigma)/denomimator)
                r_b = s + 1/self.gamma * np.log(1 + ((-1 - 2*self.q) * self.gamma * self.gamma * self.sigma * self.sigma)/denomimator)

                r = (r_a + r_b)/2 # reservation price


                #place order
                # if self.q + 0.001 > q_max:
                #     bid_order = sim.place_order(receive_ts, 0.0, 'BID', 0)
                # else:
                bid_order = sim.place_order( receive_ts, 0.001, 'BID', r_b)

                # if self.q - 0.001 < q_min:
                #     ask_order = ask_order = sim.place_order( receive_ts, 0.0, 'ASK', 0)
                # else: 
                ask_order = sim.place_order( receive_ts, 0.001, 'ASK', r_a)

                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders



        